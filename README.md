[Uploading readme.md.rtf…]()
{\rtf1\ansi\ansicpg1252\cocoartf2867
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fmodern\fcharset0 Courier;\f1\froman\fcharset0 Times-Roman;}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;}
{\*\expandedcolortbl;;\cssrgb\c0\c0\c0;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\fs26 \cf0 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 # CredRisk AI \'96 Explainable Credit Risk Analysis System\
\
CredRisk AI is an end-to-end machine learning system for predicting credit risk and explaining results using SHAP and LLMs. It combines XGBoost, FastAPI, and a chatbot interface to provide both accurate predictions and human-readable insights.\
\
\uc0\u55357 \u56599  Live Demo: https://credit-risk-detection.netlify.app/\
\
---\
\
## Features\
\
- Credit risk prediction using XGBoost (**91.34% ROC-AUC**)\
- Feature engineering (17 \uc0\u8594  23 features)\
- Explainable AI using SHAP\
- Risk grading system based on prediction score\
- LLM-powered chatbot (Groq API) for user interaction\
- Full-stack deployment (Netlify + Render)\
\
---\
\
## Tech Stack\
\
- **ML:** XGBoost, Scikit-learn, SHAP  \
- **Backend:** FastAPI (Python)  \
- **Frontend:** HTML, CSS, JavaScript  \
- **Deployment:** Netlify (Frontend), Render (Backend)  \
- **LLM:** Groq API  \
\
---\
\
## Project Structure\
\
\pard\pardeftab720\sa240\partightenfactor0

\f1\fs24 \cf0 ml-model-host/\uc0\u8232 \u9474 \u8232 \u9500 \u9472 \u9472  main.py\u8232 \u9500 \u9472 \u9472  model.pkl\u8232 \u9500 \u9472 \u9472  features.pkl\u8232 \u9500 \u9472 \u9472  encoding.pkl\u8232 \u9500 \u9472 \u9472  requirements.txt\u8232 \u9492 \u9472 \u9472  README.md\
\pard\pardeftab720\partightenfactor0

\f0\fs26 \cf0 \
\
---\
\
## Workflow\
\
1. User inputs financial data  \
2. Backend processes input (encoding + feature engineering)  \
3. XGBoost model predicts risk score  \
4. SHAP explains feature importance  \
5. LLM chatbot provides explanation  \
6. Results returned to frontend  \
\
---\
\
## Model Performance\
\
- Model: XGBoost Classifier  \
- Metric: ROC-AUC  \
- Score: **91.34%**\
\
---\
\
## Setup\
\
```bash\
git clone https://github.com/shatakshi-2905/ml-model-host.git\
cd ml-model-host\
pip install -r requirements.txt\
uvicorn main:app --reload\
}
