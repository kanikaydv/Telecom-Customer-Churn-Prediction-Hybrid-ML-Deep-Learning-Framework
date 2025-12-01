# Telecom-Customer-Churn-Prediction-Hybrid-ML-Deep-Learning-Framework
This project presents a comprehensive churn-prediction pipeline built using classical machine-learning models, deep-learning architectures, and a hybrid LSTM–Transformer + XGBoost framework.
The system is designed to analyze structured customer data from the IBM Telco Customer Churn dataset and produce actionable churn insights, model explanations, uplift estimates, and customer-level counterfactual reasoning.

->Features
🔹 Full End-to-End Churn Pipeline
Data integration (demographics, services, billing, account details)
Preprocessing (imputation, encoding, scaling, stratified splits)
Feature engineering
Model development (classical ML, deep learning, hybrid modeling)
Evaluation & interpretability modules

🔹 Models Implemented
Classical ML
Logistic Regression
SVM
KNN
Random Forest
XGBoost
LightGBM
Deep Learning
Artificial Neural Network (ANN)
1D-CNN
VGG-1D CNN
Hybrid Model
LSTM–Transformer for feature extraction + XGBoost for classification

🔹 Interpretability Tools
SHAP for global & local explanations
LIME for instance-level interpretability
Feature-importance visualizations
Probability calibration

🔹 Decision Science Layer
Uplift modeling
Counterfactual explanations
Causal inference diagnostics

->Dataset
This project uses the IBM Telco Customer Churn dataset from Kaggle.
Link: (https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset)
It contains:
7,043 samples
21 features
Mixed categorical & numerical attributes
Binary churn labels

->Project Architecture
├── data/
├── notebooks/
│   ├── preprocessing.ipynb
│   ├── classic_models.ipynb
│   ├── deep_learning.ipynb
│   ├── hybrid_lstm_transformer.ipynb
│   ├── interpretability.ipynb
│   └── uplift_counterfactuals.ipynb
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   ├── transformers.py
│   ├── evaluation.py
│   ├── interpretability.py
│   └── utils.py
└── README.md

->Tech Stack
Python 3.10
NumPy, Pandas
Scikit-learn
TensorFlow / Keras
XGBoost, LightGBM, CatBoost
Matplotlib, Seaborn
SHAP, LIME

->Key Results
Best-performing model:

--Hybrid LSTM–Transformer + XGBoost
Highest accuracy
Strongest F1-score
Best recall for high-risk churners
Captures both sequential patterns (contracts, billing history) & contextual feature relationships

--Important Features (SHAP):
Contract type
Tenure
Monthly charges
Total charges
Internet service
Payment method

--Uplift Modeling:
Identified customer groups most responsive to retention interventions
Enabled ROI-optimized targeting strategies

--Interpretability Highlights
SHAP force plots expose how contract type, billing behavior, and service configurations drive churn risk.
LIME provides instance-level explanations for individual customer predictions.
Counterfactual reasoning suggests actionable changes (e.g., adjusting contract length or billing setup).

->Future Enhancements
Real-time churn prediction
Integration with streaming data (usage logs, call logs, network metrics)
Deployable API for telecom dashboards
Lightweight Transformer models for on-device inference
Expanded causal-inference and uplift modules
