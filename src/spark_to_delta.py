from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip
import os


def spark_etl():
    """
    Spark ETL job: reads events from JSONL, transforms them,
    and writes them into Delta format.
    """

    # Ensure JAVA_HOME is set (default for Airflow image + Java 17 install)
    java_home = os.getenv("JAVA_HOME", "/usr/lib/jvm/java-17-openjdk")
    os.environ["JAVA_HOME"] = java_home

    # ✅ Build Spark session with Delta support
    builder = (
        SparkSession.builder
        .appName("RecommendationETL")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.warehouse.dir", "/opt/airflow/warehouse")
    )

    # configure_spark_with_delta_pip must wrap the builder
    spark = configure_spark_with_delta_pip(builder).getOrCreate()

    # ✅ Read JSONL events
    df = spark.read.json("/opt/airflow/data/events.jsonl")

    # Print schema for debugging
    print("✅ DataFrame Schema:")
    df.printSchema()

    # Example transformation: count by device
    if "device" in df.columns:
        counts = df.groupBy("device").count()
        print("✅ Counts by device:")
        counts.show(truncate=False)

    # ✅ Write to Delta Lake (overwrite schema to allow new session_id column)
    df.write.format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", "true") \
        .save("/opt/airflow/delta/events")

    spark.stop()
    print("✅ Spark ETL job completed successfully")
