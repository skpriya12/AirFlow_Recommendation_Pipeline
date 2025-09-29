import os
import pickle
import pandas as pd
import mlflow
import mlflow.sklearn
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import precision_at_k, recall_at_k


def train(df, test_ratio=0.2):
    """
    Temporal split: sort by user_id + timestamp, then split last N% as test.
    """
    df_sorted = df.sort_values(by=["user_id", "timestamp"])
    cutoff = int(len(df_sorted) * (1 - test_ratio))
    return df_sorted.iloc[:cutoff], df_sorted.iloc[cutoff:]


def train(**context):
    """
    Train LightFM with hyperparameter search, log results to MLflow,
    and save the best model + dataset.
    """

    # Input/output paths
    input_path = "/opt/airflow/delta/events"
    model_path = "/opt/airflow/data/model_lightfm.pkl"

    # Load events
    df = pd.read_parquet(input_path)

    # Map event_type -> numeric interaction
    def map_interaction(ev):
        if ev == "purchase":
            return 1.0
        elif ev == "add_to_cart":
            return 0.5
        else:
            return 0.0

    df["interaction"] = df["event_type"].map(map_interaction)

    # Build dataset
    dataset = Dataset()
    dataset.fit(df["user_id"].unique(), df["product_id"].unique())

    interactions_list = [
        (row.user_id, row.product_id, row.interaction) for row in df.itertuples()
    ]
    (interactions, _) = dataset.build_interactions(interactions_list)

    # --- Train/test split ---
    if "timestamp" in df.columns:
        print("⏳ Using temporal split")
        train_df, test_df = temporal_train_test_split(df, test_ratio=0.2)

        train_interactions, _ = dataset.build_interactions(
            (row.user_id, row.product_id, row.interaction) for row in train_df.itertuples()
        )
        test_interactions, _ = dataset.build_interactions(
            (row.user_id, row.product_id, row.interaction) for row in test_df.itertuples()
        )
    else:
        print("🎲 No timestamp → using random split")
        train_interactions, test_interactions = random_train_test_split(
            interactions, test_percentage=0.2, random_state=42
        )

    # 🔍 Hyperparameter search configs
    configs = [
        {"loss": "warp", "no_components": 32, "epochs": 20},
        {"loss": "warp", "no_components": 64, "epochs": 30},
        {"loss": "bpr", "no_components": 64, "epochs": 30},
        {"loss": "warp-kos", "no_components": 128, "epochs": 40},
    ]

    best_score = -1.0
    best_model = None
    best_config = None

    mlflow.set_tracking_uri("http://mlflow:5000")

    for cfg in configs:
        model = LightFM(
            loss=cfg["loss"],
            no_components=cfg["no_components"],
        )

        model.fit(train_interactions, epochs=cfg["epochs"], num_threads=4)

        # Evaluate
        prec_train = float(precision_at_k(model, train_interactions, k=5).mean())
        rec_train = float(recall_at_k(model, train_interactions, k=5).mean())
        prec_test = float(precision_at_k(model, test_interactions, k=5).mean())
        rec_test = float(recall_at_k(model, test_interactions, k=5).mean())

        # Log to MLflow
        with mlflow.start_run(
            run_name=f"lightfm_{cfg['loss']}_{cfg['no_components']}c_{cfg['epochs']}e"
        ):
            mlflow.log_params(cfg)
            mlflow.log_metrics(
                {
                    "precision_at_5_train": prec_train,
                    "recall_at_5_train": rec_train,
                    "precision_at_5_test": prec_test,
                    "recall_at_5_test": rec_test,
                }
            )
            mlflow.sklearn.log_model(model, artifact_path="lightfm_model")

        print(f"✅ Config {cfg} → Precision@5 (test): {prec_test:.4f}, Recall@5 (test): {rec_test:.4f}")

        # Track best model by test precision
        if prec_test > best_score:
            best_score = prec_test
            best_model = model
            best_config = cfg

    # Save best model + dataset + interactions_list
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump((best_model, dataset, interactions_list), f)

    print(f"🎯 Best config: {best_config} with Precision@5={best_score:.4f}")
    print(f"✅ Best model saved at {model_path}")

    # Push results to XCom (force to Python float/dict)
    ti = context["ti"]
    ti.xcom_push(key="best_config", value=best_config)
    ti.xcom_push(key="precision_at_5", value=float(best_score))

    return {"precision_at_5": float(best_score), "best_config": best_config}
