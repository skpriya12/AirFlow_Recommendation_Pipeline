from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta

# Import your pipeline functions from src (renamed modules)
from src.generate_synthetic import generate
from src.kafka_producer import publish
from src.spark_to_delta import spark_etl
from src.train_lightfm import train
from src.og_mlflow import log_metrics
from src.notify_api import notify
from src.evaluate_model import evaluate


default_args = {"owner": "airflow", "retries": 1}

with DAG(
    dag_id="recommendation_pipeline_dag",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=days_ago(1),
    catchup=False,
    tags=["recommendation"],
) as dag:

    # 1. Generate synthetic events
    generate_task = PythonOperator(
        task_id="generate_events",
        python_callable=generate,
    )

    # 2. Publish to Kafka
    publish_task = PythonOperator(
        task_id="publish_to_kafka",
        python_callable=publish,
    )

    # 3. Spark ETL
    spark_task = PythonOperator(
        task_id="spark_etl",
        python_callable=spark_etl,
    )

    # 4. Train LightFM
    train_task = PythonOperator(
        task_id="train_lightfm",
        python_callable=train,
    )

    # 5. Log metrics to MLflow (pushes run_id to XCom)
    mlflow_task = PythonOperator(
        task_id="log_mlflow",
        python_callable=log_metrics,
        retries=2,
        retry_delay=timedelta(minutes=1),
        provide_context=True,  # 👈 enables XCom push
    )
    # 6. Evaluate model (recommendations + metrics)
    evaluate_task = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate,
        provide_context=True,  # needed to pull XCom (interactions list, etc.)
    )
    # 7. Notify reco_api (pulls run_id from XCom)
    notify_task = PythonOperator(
        task_id="notify_reco_api",
        python_callable=notify,
        provide_context=True,  # 👈 enables XCom pull
    )

    # DAG order
    generate_task >> publish_task >> spark_task >> train_task >> mlflow_task >> evaluate_task >> notify_task
