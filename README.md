# 🧠 AI Project — Titanic Survivors  

## Project Overview  
- **Problem type:** Classification  
- **Business value:** The goal is to predict which passengers were more likely to survive the Titanic disaster based on demographic and socio-economic features. The model can help understand human survival patterns and factors influencing decision-making in critical situations.  
- **Data source:** [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data) that includes passenger demographics, class, fare, and survival information.  

## Architecture  
See `docs/architecture_diagram.png` and sections below.  
Main modules:  
- **ETL / Data Processing:** Kedro pipelines for cleaning, imputing, encoding, and feature scaling.  
- **Modeling:** Logistic Regression / Random Forest for classification.  
- **API Backend:** FastAPI — endpoint `/predict` returning survival probability.  
- **UI Frontend:** Streamlit — form where the user can enter passenger data and get a prediction.  
- **Automation:** Airflow DAG to run daily retraining and data validation.  
- **Deployment:** Dockerized API & UI containers.  

## How to run  
### Quickstart  
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd ai_project_teamX
kedro run
```

### Run API  
```bash
uvicorn ai_project_teamX.api.main:app --reload --port 8000
```

### Run UI  
```bash
streamlit run ai_project_teamX/ui/app.py --server.port 8501
```

## Team  
| Name          | Role | GitHub      |
|---------------|------|-------------|
| Mariia Shulyk | Team Lead| MariaShulyk |

## Links  
- **Repo:** *https://github.com/PJATK-ASI-2024/ai_project_titanic*  
- **Project Board:** *https://github.com/orgs/PJATK-ASI-2024/projects/219/views/1*
- **Docs / Architecture:**   