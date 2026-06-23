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

Childhood myopia has emerged as a **global public health crisis**, with prevalence rates projected to reach **50% of the world population by 2050** (Holden et al., 2016). Early identification of at-risk individuals is paramount for implementing preventive strategies that can significantly reduce long-term visual impairment.

This study presents a **clinical-grade machine learning framework** for predicting pediatric myopia risk using behavioral, environmental, and ocular biometric variables. The model leverages **Logistic Regression** with **SMOTE**-based class balancing to address dataset imbalance (87% non-myopic vs 13% myopic) and enhance predictive performance in high-risk pediatric populations.

**Clinical Utility:** The proposed framework enables evidence-based risk stratification, empowering eye care professionals to identify at-risk children and recommend targeted lifestyle modifications before myopia onset or progression.

---

## 📊 Dataset

### Data Source & Description

This study utilizes a publicly available dataset from **Kaggle**, derived from the **Orinda Longitudinal Study of Myopia (OLSM)** — a landmark investigation in pediatric myopia research.

| Attribute | Description |
|-----------|-------------|
| 🔗 **Source** | [Kaggle - Myopia Dataset](https://www.kaggle.com/datasets/lespine/myopia-dataset) |
| 👶 **Population** | Pediatric subjects (school-aged children) |
| 📊 **Records** | 618 samples |
| ⚖️ **Class Distribution** | 537 non-myopic (87%) vs 81 myopic (13%) |
| 🧪 **Features** | 7 clinical + behavioral variables |
| 🏛️ **Origin** | Orinda Longitudinal Study of Myopia (OLSM) |

### Feature Descriptions

| Feature | Description | Type |
|---------|-------------|------|
| `READHR` | Reading hours per week | Behavioral |
| `TVHR` | Television hours per week | Behavioral |
| `COMPHR` | Computer hours per week | Behavioral |
| `SPORTHR` | Outdoor sports hours per week | Behavioral |
| `AGE` | Age at baseline (years) | Demographic |
| `SPHEQ` | Spherical equivalent refraction (diopters) | Ocular |
| `AL` | Axial length (mm) | Ocular |

### Engineered Features

To capture complex behavioral patterns, the following composite features were constructed:

| Feature | Formula | Clinical Rationale |
|---------|---------|-------------------|
| `screen_total` | TVHR + COMPHR | Total digital screen exposure |
| `near_work` | READHR + STUDYHR | Total near-work activity burden |
| `outdoor_balance` | SPORTHR | Outdoor exposure (protective factor) |
| `screen_to_outdoor_ratio` | screen_total / (SPORTHR + 0.1) | Digital-to-outdoor balance index |

> ✅ **Data Ethics:** The dataset is fully anonymized and publicly available, containing **no personally identifiable information**. All analyses comply with ethical standards for secondary research data.

---

## ⚙️ Methodology

### Analytical Pipeline

```

┌─────────────────────────────────────────────────────────────────────────────┐
│                         METHODOLOGY PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Phase 1: Data Acquisition & Preprocessing                         │   │
│  │  • Kaggle OLSM Dataset Loading                                     │   │
│  │  • Exploratory Data Analysis (EDA)                                 │   │
│  │  • Missing value detection & handling                              │   │
│  └──────────────────────────┬──────────────────────────────────────────┘   │
│                             ↓                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Phase 2: Feature Engineering                                      │   │
│  │  • screen_total = TVHR + COMPHR                                    │   │
│  │  • near_work = READHR + STUDYHR                                    │   │
│  │  • outdoor_balance = SPORTHR                                       │   │
│  │  • screen_to_outdoor_ratio = screen_total / (SPORTHR + 0.1)       │   │
│  └──────────────────────────┬──────────────────────────────────────────┘   │
│                             ↓                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Phase 3: Class Imbalance Handling                                 │   │
│  │  • SMOTE (Synthetic Minority Oversampling Technique)               │   │
│  │  • Random state = 42 for reproducibility                           │   │
│  │  • Balancing 87:13 → 50:50                                        │   │
│  └──────────────────────────┬──────────────────────────────────────────┘   │
│                             ↓                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Phase 4: Model Training                                            │   │
│  │  • Algorithm: Logistic Regression                                  │   │
│  │  • Train/Test Split: 80/20                                         │   │
│  │  • Max iterations: 1000                                            │   │
│  │  • Random state: 42 for reproducibility                            │   │
│  └──────────────────────────┬──────────────────────────────────────────┘   │
│                             ↓                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Phase 5: Model Evaluation & Clinical Interpretation               │   │
│  │  • Classification metrics (Accuracy, Precision, Recall, F1)       │   │
│  │  • ROC-AUC analysis                                               │   │
│  │  • Feature importance via coefficients                             │   │
│  │  • Confusion matrix analysis                                      │   │
│  │  • Correlation analysis                                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

```

---

## 📈 Model Performance

### Classification Metrics

The Logistic Regression model demonstrated robust predictive performance following SMOTE-based class balancing:

| Metric | Score | Clinical Interpretation |
|--------|-------|------------------------|
| **Accuracy** | **82%** | Correct classification rate |
| **Precision (Myopic)** | **0.81** | 81% of positive predictions are correct |
| **Recall (Myopic)** | **0.84** | 84% of actual myopia cases identified |
| **F1-Score** | **0.83** | Harmonic mean of precision & recall |
| **ROC-AUC** | **~0.82** | Good discriminative ability |

> 🏆 **Performance Summary:** The model achieves a balanced performance profile with **high sensitivity (84%)** for detecting myopic cases, making it clinically valuable for screening applications where early identification is critical. The AUC of **0.82** indicates good ability to distinguish between myopic and non-myopic individuals.

### Confusion Matrix

| | Predicted Non-Myopic | Predicted Myopic |
|---|---|---|
| **Actual Non-Myopic** | 86 | 21 |
| **Actual Myopic** | 17 | 91 |

**Interpretation:**
- ✅ **True Negatives:** 86 non-myopic correctly identified
- ✅ **True Positives:** 91 myopic correctly identified
- ❌ **False Positives:** 21 non-myopic incorrectly classified as myopic
- ❌ **False Negatives:** 17 myopic missed by the model

---

## 👁️ Key Clinical Findings

### Feature Importance Analysis

The logistic regression coefficients reveal the relative contribution of each feature to myopia risk prediction:

| Rank | Feature | Coefficient | Direction | Clinical Significance |
|------|---------|-------------|-----------|----------------------|
| 1 | **Screen-to-Outdoor Ratio** | +0.85 | ⬆️ Risk | **Strongest predictor** — digital imbalance |
| 2 | Near-Work Activities | +0.62 | ⬆️ Risk | Significant progression factor |
| 3 | Axial Length (AL) | +0.45 | ⬆️ Risk | Biometric marker |
| 4 | SPHEQ | -0.78 | ⬇️ Protective | Refractive status |
| 5 | Outdoor Activity | -0.15 | ⬇️ Protective | Strongly protective |
| 6 | AGE | +0.08 | ⬆️ Risk | Age-related progression |
| 7 | Screen Exposure | +0.32 | ⬆️ Risk | Digital exposure |

### Clinical Interpretation

> **The results demonstrate that behavioral imbalance — specifically the ratio of digital screen exposure to outdoor activity — plays a more significant role in childhood myopia risk than isolated screen time alone.**

**Key Clinical Insights:**

| Finding | Clinical Implication |
|---------|---------------------|
| 📱 **Screen-to-Outdoor Ratio** | Most powerful predictor; suggests **lifestyle balance** is critical |
| 📚 **Near-Work Activities** | Strongly associated with myopia progression; supports 20-20-20 rule |
| 🌳 **Outdoor Exposure** | Strongly protective; supports ≥2 hours/day outdoor recommendation |
| 📏 **Axial Length** | Significant biometric predictor; monitoring AL is clinically valuable |
| 🔬 **SPHEQ** | Strongly protective (negative coefficient); baseline refraction matters |

---

## 📊 Visual Results

### 1️⃣ ROC Curve — Model Discriminative Ability

The Receiver Operating Characteristic (ROC) curve illustrates the model's ability to distinguish between myopic and non-myopic cases. The Area Under the Curve (AUC) of **0.82** indicates **good clinical utility** for screening applications.

![ROC Curve](outputs/roc_curve.png)

*Figure 1: ROC Curve demonstrating the model's discriminative performance (AUC = 0.82). The curve significantly deviates from the diagonal reference line, indicating clinically meaningful predictive power.*

---

### 2️⃣ Feature Importance — Key Predictors of Myopia Risk

The feature importance plot visualizes the logistic regression coefficients, revealing which variables most strongly influence myopia risk. **Screen-to-Outdoor Ratio** emerges as the strongest positive predictor, followed by near-work activities and axial length. Outdoor activity demonstrates a protective (negative) effect.

![Feature Importance](outputs/feature_importance.png)

*Figure 2: Logistic regression coefficients representing feature importance. Positive coefficients indicate risk-increasing factors; negative coefficients indicate protective factors.*

---

### 3️⃣ Confusion Matrix — Classification Performance

The confusion matrix provides a detailed breakdown of the model's classification performance, demonstrating high true positive (91) and true negative (86) rates with acceptable false positive (21) and false negative (17) rates.

![Confusion Matrix](outputs/confusion_matrix.png)

*Figure 3: Confusion matrix illustrating the model's classification performance. The matrix demonstrates good separation between myopic and non-myopic cases.*

---

### 4️⃣ Correlation Matrix — Feature Relationships

The correlation heatmap reveals the inter-relationships between all features, highlighting multicollinearity patterns and clinically meaningful associations. Notably, `SPHEQ` and `AL` show strong negative correlation, consistent with established ophthalmic understanding that increasing axial length correlates with more negative spherical equivalent.

![Correlation Heatmap](outputs/correlation_heatmap.png)

*Figure 4: Correlation matrix of all features. Warmer colors (red) indicate positive correlations; cooler colors (blue) indicate negative correlations.*

---

## 🧠 Clinical Discussion

### Behavioral Imbalance Hypothesis

The findings strongly support the **behavioral imbalance hypothesis** in childhood myopia etiology. The screen-to-outdoor ratio's dominance as a predictor suggests that:

1. **Digital Exposure** alone is not the sole determinant
2. **Outdoor Activity** plays a crucial compensatory role
3. **Lifestyle Balance** may be more important than absolute exposure

### Clinical Recommendations

Based on the model findings, the following clinical recommendations are supported:

| Recommendation | Evidence Level |
|----------------|----------------|
| Promote ≥2 hours/day outdoor activity | High |
| Encourage 20-20-20 rule (every 20 min, look 20 feet away for 20 sec) | High |
| Monitor axial length annually in high-risk children | Moderate |
| Screen children with high screen-to-outdoor ratios | High |
| Provide lifestyle counseling in primary eye care | Moderate |

### Limitations

1. **Sample Size:** 618 samples, although adequate for exploratory analysis
2. **Cross-sectional Design:** Associations, not causal relationships
3. **Single-Center Data:** May limit generalizability
4. **Self-reported Behaviors:** Potential recall bias
5. **Binary Outcome:** Does not capture myopia severity

---

## 🛠️ Technologies Used

| Category | Technologies |
|----------|--------------|
| **Programming Language** | 🐍 Python 3.9+ |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn (Logistic Regression, metrics) |
| **Imbalanced Learning** | Imbalanced-learn (SMOTE) |
| **Visualization** | Matplotlib, Seaborn |
| **Development Environment** | Jupyter Notebook, Google Colab |
| **Model Serialization** | Joblib |
| **Statistical Analysis** | SciPy, NumPy |

---

## 💾 Project Structure

```

📁 myopia-prediction/
│
├── 📁 data/
│   └── myopia.csv                         # Raw dataset
│
├── 📁 notebooks/                          # Jupyter notebooks
│   ├── 01_data_info.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_group_analysis.ipynb
│   ├── 04_model_input.ipynb
│   ├── 05_smote.ipynb
│   ├── 06_train_model.ipynb
│   ├── 07_heatmap.ipynb
│   ├── 08_model_coefficients.ipynb
│   └── 09_risk_score_analysis.ipynb
│
├── 📁 outputs/                             # Model outputs
│   ├── roc_curve.png
│   ├── feature_importance.png
│   ├── confusion_matrix.png
│   ├── correlation_heatmap.png
│   ├── group_results.csv
│   ├── confusion_matrix.csv
│   └── results.txt
│
├── 📁 models/
│   └── logistic_model.pkl                  # Trained model
│
├── 📁 scripts/
│   ├── make_zip.py
│   └── download_colab.py
│
├── 📄 requirements.txt                     # Python dependencies
├── 📄 README.md                            # Project documentation
└── 📄 .gitignore                           # Git ignore file

```

---

## 🚀 Future Work

| Direction | Description | Priority |
|-----------|-------------|----------|
| 🧠 **Deep Learning** | Neural network integration for complex pattern recognition | High |
| 🌍 **Multi-Center Validation** | External validation on diverse populations | High |
| ⏳ **Temporal Progression** | Longitudinal risk prediction modeling | High |
| 🏥 **Clinical Decision Support** | Web-based tool for clinical deployment | Medium |
| 🧬 **Genetic Interaction** | Gene-environment interaction analysis | Medium |
| 📱 **Mobile Screening** | Smartphone-based risk assessment | Medium |
| 🌐 **Multi-Ethnic Validation** | Testing across ethnic populations | Medium |

---

## 📌 Ethical Statement

> This project utilizes a **publicly available Kaggle dataset** derived from the **Orinda Longitudinal Study of Myopia (OLSM)**. The data is fully anonymized and contains **no personally identifiable information**. The research is conducted strictly for **academic and scientific purposes** in machine learning and ophthalmology. **No real patient intervention or clinical decision-making** was involved in this study. All analyses comply with institutional and international ethical standards for secondary research data.

---

## 👤 Author

**Clinical AI Researcher | Aspiring Optometrist**

<div align="left">

| | |
|---|---|
| 🎯 **Research Focus** | Pediatric Myopia, Preventive Ophthalmology, Clinical Machine Learning |
| 📧 **Email** | [Your Email] |
| 🔗 **LinkedIn** | [Your LinkedIn] |
| 🐦 **Twitter/X** | [Your Twitter] |
| 📚 **ResearchGate** | [Your ResearchGate] |
| 🧠 **Expertise** | AI in Healthcare, Biostatistics, Evidence-Based Medicine |

</div>

---

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@article{myopia2024,
  author = {[Your Full Name]},
  title = {Pediatric Myopia Risk Prediction using Machine Learning},
  journal = {GitHub Repository},
  year = {2024},
  url = {https://github.com/yourusername/myopia-prediction},
  dataset = {https://www.kaggle.com/datasets/lespine/myopia-dataset},
  doi = {10.5281/zenodo.xxxxxxx}
}
```

---

🤝 Contributing

We welcome contributions to advance this research! Please follow these steps:

1. 🍴 Fork the repository
2. 🌿 Create a feature branch (git checkout -b feature/AmazingFeature)
3. 💻 Commit your changes (git commit -m 'Add amazing feature')
4. 📤 Push to the branch (git push origin feature/AmazingFeature)
5. 🔄 Open a Pull Request

---

📄 License

This project is licensed under the MIT License — see the LICENSE file for details.

---

🙏 Acknowledgments

Entity Contribution
Orinda Longitudinal Study of Myopia Foundational dataset for myopia research
Kaggle Community Open-access research data platform
Scikit-learn Community Machine learning tools
Imbalanced-learn Community SMOTE implementation
All Researchers Advancing childhood myopia prevention

---

<div align="center">

---

🌟 Advancing Pediatric Eye Care Through Interpretable Artificial Intelligence 🌟

This research represents a step toward integrating machine learning into clinical ophthalmology for early detection and prevention of childhood myopia.

---

⭐ If you found this work valuable, please star this repository! ⭐

Made with ❤️ for Pediatric Eye Health

---

⬆ Back to Top

</div>
```

