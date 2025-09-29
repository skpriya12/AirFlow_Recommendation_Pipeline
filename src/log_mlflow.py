import mlflow
import mlflow.sklearn
import pickle
import os
from datetime import datetime


def log_metrics(**context):
    """
    Loads the trained LightFM model, logs it to MLflow, and pushes run_id to XCom.
    Pulls real evaluation metrics from the evaluate task (via XCom).
    """
    model_path = "/opt/airflow/data/model_lightfm.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"❌ Model file not found at {model_path}. Did the train task run successfully?"
        )

    # Load everything that was pickled (model, dataset, interactions_list)
    with open(model_path, "rb") as f:
        loaded = pickle.load(f)

    if isinstance(loaded, tuple) and len(loaded) == 3:
        model, dataset, _ = loaded
    elif isinstance(loaded, tuple) and len(loaded) == 2:
        model, dataset = loaded
    else:
        raise ValueError("❌ Unexpected pickle format in model_lightfm.pkl")

    # 🔑 Point to MLflow tracking server
    mlflow.set_tracking_uri("http://mlflow:5000")

    ti = context["ti"]

    # Get training params from XCom
    train_params = ti.xcom_pull(task_ids="train_lightfm", key="train_params") or {}

    params = {
        "model": "LightFM",
        "loss": model.loss,
        "no_components": getattr(model, "no_components", None),
        "epochs": train_params.get("epochs"),
        "num_threads": train_params.get("num_threads"),
    }

    # ✅ Get real metrics from evaluate task
    precision = ti.xcom_pull(task_ids="evaluate_model", key="precision_at_5")
    recall = ti.xcom_pull(task_ids="evaluate_model", key="recall_at_5")

    metrics = {
        "precision_at_5": precision,
        "recall_at_5": recall,
    }

    with mlflow.start_run(
        run_name=f"lightfm_reco_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ) as run:
        # Log params
        for k, v in params.items():
            if v is not None:
                mlflow.log_param(k, v)

        # Log metrics
        for k, v in metrics.items():
            if v is not None:
                mlflow.log_metric(k, v)

        # Save model
        mlflow.sklearn.log_model(model, artifact_path="lightfm_model")

        run_id = run.info.run_id
        print(f"✅ Logged LightFM model and metrics to MLflow (run_id={run_id})")

        ti.xcom_push(key="mlflow_run_id", value=run_id)

    return run_id
