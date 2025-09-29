# AirFlow_Recommendation_Pipeline
This project builds a recommendation pipeline with Apache Airflow, LightFM, and MLflow. It ingests user–item events from Delta tables, processes interactions, trains models with hyperparameter tuning, logs precision@k and recall@k to MLflow, selects the best config, and evaluates before deployment.
