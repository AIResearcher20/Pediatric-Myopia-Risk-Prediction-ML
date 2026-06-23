# 👁️ Pediatric Myopia Risk Prediction using Machine Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2+-orange?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-1.5+-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?logo=numpy&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-3776AB?logo=python&logoColor=white)
![SMOTE](https://img.shields.io/badge/SMOTE-Imbalanced--learn-9B59B6?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success)
![Domain](https://img.shields.io/badge/Domain-Ophthalmology-red)
![Research-Level](https://img.shields.io/badge/Level-Clinical%20AI%20Research-critical)

</div>

---

## 🧠 Abstract

This project presents a **clinical-grade machine learning framework** for predicting childhood myopia risk using behavioral, environmental, and ocular biometric variables. The model focuses on interpretability, clinical relevance, and early risk stratification for preventive ophthalmic care.

---

## 📊 Dataset

This study uses a publicly available dataset from Kaggle, originally derived from the **Orinda Longitudinal Study of Myopia (OLSM)**.

### 📌 Dataset Source
- 🔗 Kaggle Dataset: https://www.kaggle.com/datasets/lespine/myopia-dataset  
- 👶 Pediatric population dataset  
- 📊 618 clinical records  
- ⚖️ Binary outcome: Myopic vs Non-Myopic  
- 🧪 Features: behavioral + demographic + ocular measurements  

### 📥 How to Use Dataset

```bash
# Download directly using Kaggle API
kaggle datasets download -d lespine/myopia-dataset

or in Python:

import opendatasets as od
od.download("https://www.kaggle.com/datasets/lespine/myopia-dataset")


---

⚙️ Methodology Pipeline

Data Collection (Kaggle OLSM-derived dataset)
        ↓
Data Cleaning & Preprocessing
        ↓
Exploratory Data Analysis (EDA)
        ↓
Feature Engineering
    - screen_total
    - near_work
    - outdoor_balance
    - screen_to_outdoor_ratio
        ↓
Class Imbalance Handling (SMOTE)
        ↓
Logistic Regression Model Training
        ↓
Evaluation (Train/Test Split 80/20)
        ↓
Clinical Interpretation & Visualization


---

📈 Model Performance

Metric	Score

Accuracy	82%
Precision	0.81
Recall	0.84
F1-score	0.83
ROC-AUC	~0.82



---

👁️ Clinical Interpretation

The model indicates that behavioral imbalance between near-work, screen exposure, and outdoor activity plays a major role in pediatric myopia risk.

Key clinical insights:

📱 High screen exposure increases risk

📚 Near-work activities contribute significantly

🌳 Outdoor activity is strongly protective

⚠️ Screen-to-outdoor ratio is the strongest predictor



---

📊 Visual Results

1️⃣ ROC Curve (Model Performance)




---

2️⃣ Feature Importance (Logistic Regression Coefficients)




---

3️⃣ Correlation Heatmap (Feature Relationships)




---

🛠️ Technologies Used

Python

Pandas

NumPy

Scikit-learn

Seaborn

Matplotlib

Imbalanced-learn (SMOTE)

Jupyter Notebook

Joblib



---

💾 Project Structure

myopia-project/
│
├── data/
│   └── myopia.csv
│
├── outputs/
│   ├── roc_curve.png
│   ├── feature_importance.png
│   └── correlation_heatmap.png
│
├── models/
│   └── logistic_model.pkl
│
├── notebooks/
├── src/
└── README.md


---

🚀 Future Work

Deep learning-based risk prediction models

Multi-center validation studies

Longitudinal progression modeling

Clinical decision support system deployment

Integration with ophthalmic screening tools



---

📌 Ethical Statement

This project uses a publicly available and fully anonymized Kaggle dataset derived from the Orinda Longitudinal Study of Myopia (OLSM). No personally identifiable information is included. The study is conducted strictly for academic and machine learning research purposes in ophthalmology.


---

👤 Author

Clinical AI Researcher | Aspiring Optometrist

Focus Areas:

Pediatric Myopia

Preventive Ophthalmology

Clinical Machine Learning

Evidence-Based Eye Care



---

⭐ Acknowledgment

Orinda Longitudinal Study of Myopia (OLSM) and Kaggle community for providing open-access research data.


---

<div align="center">Advancing Pediatric Eye Care Through Interpretable Artificial Intelligence

</div>
```
---
