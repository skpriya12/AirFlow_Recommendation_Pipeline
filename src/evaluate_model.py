import pickle
import numpy as np
from lightfm.evaluation import precision_at_k, recall_at_k


def evaluate(**context):
    """
    Load trained LightFM model + dataset (+ optional interactions_list),
    compute recommendations + metrics, and push to XCom.
    Works for both user-level and session-level entity IDs.
    """
    model_path = "/opt/airflow/data/model_lightfm.pkl"
    with open(model_path, "rb") as f:
        obj = pickle.load(f)

    # Handle whether train() saved (model, dataset) or (model, dataset, interactions_list)
    if isinstance(obj, tuple) and len(obj) == 3:
        model, dataset, saved_interactions_list = obj
    elif isinstance(obj, tuple) and len(obj) == 2:
        model, dataset = obj
        saved_interactions_list = None
    else:
        raise ValueError("❌ Unexpected model.pkl contents")

    print("✅ Loaded LightFM model and dataset")

    # Get interactions_list (prefer XCom, fallback to saved version)
    ti = context["ti"]
    interactions_list = ti.xcom_pull(task_ids="train_lightfm", key="interactions_list")
    if not interactions_list and saved_interactions_list:
        interactions_list = saved_interactions_list

    if not interactions_list:
        raise ValueError("❌ No interactions_list found in XCom or model.pkl")

    # Build interactions matrix
    (interactions, _) = dataset.build_interactions(interactions_list)

    # Show example recommendations for first 4 entities
    all_entities = list(dataset.mapping()[0].keys())
    print("\n🎯 Example Recommendations:")
    for entity in all_entities[:4]:
        recs = recommend_top_n(model, dataset, entity, n=3)
        print(f"{entity} → {recs}")

    # Compute metrics
    prec = precision_at_k(model, interactions, k=5).mean()
    rec = recall_at_k(model, interactions, k=5).mean()

    # Get best config from training XCom
    best_config = ti.xcom_pull(task_ids="train_lightfm", key="best_config")

    print(f"\n📊 Precision@5: {prec:.4f}")
    print(f"📊 Recall@5:    {rec:.4f}")
    if best_config:
        print(f"⚙️ Evaluated with config: {best_config}")

    # Push metrics to XCom for MLflow logging
    ti.xcom_push(key="precision_at_5", value=float(prec))
    ti.xcom_push(key="recall_at_5", value=float(rec))
    if best_config:
        ti.xcom_push(key="best_config", value=best_config)

    return {
        "precision_at_5": float(prec),
        "recall_at_5": float(rec),
        "best_config": best_config,
    }


def recommend_top_n(model, dataset, entity_id, n=5):
    """
    Recommend top-N items for a given user_id or session_id.
    """
    entity_mapping, _, item_mapping, _ = dataset.mapping()

    if entity_id not in entity_mapping:
        raise ValueError(f"Entity {entity_id} not found in dataset mapping")

    entity_index = entity_mapping[entity_id]
    item_ids = list(item_mapping.keys())
    item_indices = np.array(list(item_mapping.values()))

    scores = model.predict(entity_index, item_indices)
    top_indices = np.argsort(-scores)[:n]

    return [item_ids[i] for i in top_indices]
