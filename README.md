# 👁️ Pediatric Myopia Risk Prediction using Machine Learnin

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2%2B-orange?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-1.5%2B-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7%2B-11557c?logo=plotly&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12%2B-3776AB?logo=python&logoColor=white)
![SMOTE](https://img.shields.io/badge/SMOTE-Imbalanced--learn-9B59B6?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-success)
---

## 🧠 Abstract

This study presents a clinical machine learning framework for predicting the risk of childhood myopia using behavioral, environmental, and ocular biometric variables. The model integrates **Logistic Regression** with **SMOTE**-based class balancing to address dataset imbalance and enhance predictive performance in high-risk pediatric populations.

**Key clinical utility:** Early identification of at-risk individuals enables preventive interventions that can significantly reduce long-term visual impairment.

---

## 👁️ Clinical Motivation

Childhood myopia is a rapidly increasing **global public health concern**, with prevalence rates projected to reach **50% of the world population by 2050** (Holden et al., 2016). 

This project aims to support **evidence-based clinical decision-making** through interpretable machine learning models that empower eye care professionals to:
- Identify at-risk children earlier
- Recommend targeted lifestyle modifications
- Monitor progression risk over time

---

## 📊 Dataset

| Feature | Description |
|---------|-------------|
| 👶 **Samples** | 618 pediatric records |
| 📌 **Features** | 7 clinical + behavioral variables |
| ⚖️ **Class Distribution** | 537 non-myopic (87%) vs 81 myopic (13%) |
| 🧪 **Source** | OLSM (Orinda Longitudinal Study of Myopia) |

**Features used:**
- `READHR` - Reading hours per week
- `TVHR` - Television hours per week
- `COMPHR` - Computer hours per week
- `SPORTHR` - Outdoor sports hours per week
- `AGE` - Age at baseline
- `SPHEQ` - Spherical equivalent refraction
- `AL` - Axial length (mm)

---

## ⚙️ Methodology

```

┌─────────────────────────────────────────────────────────────┐
│                  METHODOLOGY PIPELINE                      │
├─────────────────────────────────────────────────────────────┤
│  1. Raw Data (myopia.csv)                                  │
│         ↓                                                  │
│  2. Feature Engineering                                    │
│     • screen_total = TVHR + COMPHR                        │
│     • near_work = READHR + STUDYHR                        │
│     • outdoor_balance = SPORTHR                           │
│     • screen_to_outdoor_ratio = screen_total/(outdoor+0.1)│
│         ↓                                                  │
│  3. SMOTE Oversampling (balancing minority class)         │
│         ↓                                                  │
│  4. Train/Test Split (80/20)                               │
│         ↓                                                  │
│  5. Logistic Regression (max_iter=1000)                    │
│         ↓                                                  │
│  6. Evaluation & Visualization                              │
└─────────────────────────────────────────────────────────────┘

```

---

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 82% |
| **Precision (Myopia)** | 0.81 |
| **Recall (Myopia)** | 0.84 |
| **F1-Score** | 0.83 |
| **AUC** | ~0.82 |

---

## 👁️ Key Clinical Findings

| Finding | Interpretation |
|---------|----------------|
| 📱 **Screen Exposure** | Higher screen time → increased myopia risk |
| 📚 **Near-work Activities** | Strongly associated with myopia progression |
| 🌳 **Outdoor Exposure** | Strongly protective against myopia |
| ⚠️ **Screen-to-Outdoor Ratio** | The **most powerful** predictive indicator |

> **Clinical Insight:** Lifestyle imbalance, rather than isolated screen time, is a more robust determinant of pediatric myopia risk. This supports the need for **integrated behavioral screening** in primary eye care.

---

## 📊 Visual Outputs

| Output | Description |
|--------|-------------|
| 📈 **ROC Curve** | Model discriminative ability (AUC ~0.82) |
| 🔢 **Confusion Matrix** | Classification performance visualization |
| 📊 **Feature Importance** | Logistic regression coefficients |
| 🌡️ **Correlation Heatmap** | Feature inter-relationships |
| 📦 **Risk Score Distribution** | Myopia risk distribution across groups |

---

## 🛠️ Tech Stack

```python
# Core Libraries
import pandas as pd          # Data manipulation
import numpy as np           # Numerical operations

# Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Imbalanced Learning
from imblearn.over_sampling import SMOTE

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Model Serialization
import joblib
```

---

💾 Outputs

```
📁 outputs/
├── group_results.csv           # Feature averages by myopia status
├── confusion_matrix.csv        # Confusion matrix values
├── roc_curve.png              # ROC curve plot
├── feature_importance.png     # Feature importance bar chart
└── results.txt                # Summary statistics

📁 models/
└── logistic_model.pkl         # Trained Logistic Regression model
```

---

🚀 Future Work

· 🔬 Deep Learning Integration - Neural networks for complex pattern recognition
· 🌍 Multi-center Validation - External validation on diverse populations
· ⏳ Temporal Progression Modeling - Longitudinal risk prediction
· 🏥 Clinical Decision Support Tool - Web-based deployment for clinicians
· 🧬 Genetic Interaction Analysis - Gene-environment interactions in myopia

---

📌 Ethical Statement

This project uses anonymized research data from the Orinda Longitudinal Study of Myopia (OLSM) and is intended for academic and clinical research purposes only. No identifiable patient data is included. All analyses comply with ethical standards for secondary research data.

---

👤 Author
Sepideh Moafi 

Aspiring Optometrist | Clinical AI Researcher

· 🎯 Focus: Pediatric Myopia • AI in Ophthalmology • Preventive Eye Care


---

📝 Citation

If you use this work, please cite:

```bibtex
@article{myopia2024,
  author = {[Your Name]},
  title = {Pediatric Myopia Risk Prediction using Machine Learning},
  year = {2024},
  journal = {GitHub Repository},
  url = {https://github.com/yourusername/myopia-prediction}
}
```

---

🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (git checkout -b feature/AmazingFeature)
3. Commit your changes (git commit -m 'Add some AmazingFeature')
4. Push to the branch (git push origin feature/AmazingFeature)
5. Open a Pull Request

---

📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

⭐ If you found this project useful, please star it on GitHub!

```

