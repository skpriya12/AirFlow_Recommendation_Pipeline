# AirFlow_Recommendation_Pipeline

## 📌 Project Overview
AirFlow_Recommendation_Pipeline is a production-ready recommendation system pipeline built with **Apache Airflow** for orchestration, **LightFM** for training hybrid recommendation models, and **MLflow** for experiment tracking.  
It ingests user–item interaction data from **Delta tables**, processes interactions, trains and tunes models, logs performance metrics, and deploys the best model.

---

## ⚙️ Features
- **Automated Orchestration**: Managed with Apache Airflow DAGs.  
- **Model Training**: Uses LightFM with multiple losses (warp, bpr, warp-kos).  
- **Hyperparameter Tuning**: Iterates configs (components, epochs, learning rate, regularization).  
- **Experiment Tracking**: Logs metrics and parameters to MLflow.  
- **Evaluation**: Computes `precision@k` and `recall@k`.  
- **Deployment Ready**: Saves best model and dataset for downstream services.  

---

## 📂 Pipeline Flow
1. **Data Ingestion** – Load user–item events from Delta tables.  
2. **Preprocessing** – Map events (`purchase`, `add_to_cart`) to interaction weights.  
3. **Training & Tuning** – Train LightFM models on train/test splits with multiple configs.  
4. **Experiment Logging** – Log parameters and metrics to MLflow.  
5. **Model Selection** – Choose best configuration by test precision@5.  
6. **Evaluation** – Generate recommendations and compute metrics.  
7. **Save Artifacts** – Persist best model, dataset, and interactions.  

---

## 🚀 Tech Stack
- **Apache Airflow** – Workflow orchestration  
- **LightFM** – Recommendation model training  
- **MLflow** – Experiment tracking and model registry  
- **Delta Lake** – Source of user–item events  
- **Python 3.11** – Implementation language  

---

## 📊 Metrics
The pipeline evaluates models with:
- **Precision@k** – Fraction of recommended items that are relevant.  
- **Recall@k** – Fraction of relevant items successfully recommended.  

---

## ▶️ Getting Started

### Prerequisites
- Python 3.11+
- Apache Airflow
- MLflow server running (`mlflow ui --host 0.0.0.0 --port 5000`)
- Delta Lake tables with `user_id`, `product_id`, `event_type`

### Installation
```bash
git clone https://github.com/your-username/AirFlow_Recommendation_Pipeline.git
cd AirFlow_Recommendation_Pipeline
pip install -r requirements.txt
