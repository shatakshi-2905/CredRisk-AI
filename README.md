# CredRisk AI – Explainable Credit Risk Analysis System

CredRisk AI is an end-to-end machine learning system for predicting credit risk and explaining results using SHAP and LLMs. It combines XGBoost, FastAPI, and a chatbot interface to provide both accurate predictions and human-readable insights.

🔗 Live Demo: https://credrisk-ai.netlify.app/

---

## Features

- Credit risk prediction using XGBoost (**91.34% ROC-AUC**)
- Feature engineering (17 → 23 features)
- Explainable AI using SHAP
- Risk grading system based on prediction score
- LLM-powered chatbot (Groq API) for user interaction
- Full-stack deployment (Netlify + Render)

---

## Tech Stack

- **ML:** XGBoost, Scikit-learn, SHAP  
- **Backend:** FastAPI (Python)  
- **Frontend:** HTML, CSS, JavaScript  
- **Deployment:** Netlify (Frontend), Render (Backend)  
- **LLM:** Groq API  

---

## Project Structure

ml-model-host/ │ ├── main.py ├── model.pkl ├── features.pkl ├── encoding.pkl ├── requirements.txt └── README.md


---

## Workflow

1. User inputs financial data  
2. Backend processes input (encoding + feature engineering)  
3. XGBoost model predicts risk score  
4. SHAP explains feature importance  
5. LLM chatbot provides explanation  
6. Results returned to frontend  

---

## Model Performance

- Model: XGBoost Classifier  
- Metric: ROC-AUC  
- Score: **91.34%**

---

## Setup

```bash
git clone https://github.com/shatakshi-2905/ml-model-host.git
cd ml-model-host
pip install -r requirements.txt
uvicorn main:app --reload
