# 🏥 Precision Health: Breast Cancer Diagnostic Classification

> **High-reliability Machine Learning pipeline** designed to classify breast tumors as Malignant or Benign with **97.48% Cross-Validation accuracy**. This project demonstrates a rigorous approach to algorithm selection and hyperparameter optimization for clinical decision support.

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

---

## 🎯 Executive Summary
The primary goal was to develop a diagnostic aid tool using Fine-Needle Aspirate (FNA) biopsy data. By comparing multiple architectures and performing exhaustive hyperparameter tuning, the project delivers a model that balances high precision with the critical need for high recall in oncology.

---

## 🏗️ Clinical Data Pipeline
The data follows a standardized medical-grade preprocessing flow:
1. **Cleaning:** Removal of non-informative features (ID, empty columns).
2. **Feature Engineering:** Analysis of 30 clinical metrics (radius, texture, perimeter, etc.).
3. **Scaling:** Implementation of `StandardScaler` to normalize feature influence.
4. **Optimization:** 5-fold Cross-Validation with `GridSearchCV` across 4 different architectures.



---

## 📊 Benchmark & Model Selection
Instead of using default settings, I conducted a systematic comparison of the **best-tuned versions** of each algorithm:

| Algorithm | Best CV Accuracy | Key Parameters |
| :--- | :---: | :--- |
| **Logistic Regression (L2)** | **97.48%** | `C=1.0` |
| KNN | 96.98% | `k=7` |
| SVM (Linear) | 96.72% | `C=0.1` |
| Random Forest | 95.22% | `n_estimators=50` |

### Why Logistic Regression?
While non-linear models are often favored, **Logistic Regression** provided the best balance of performance and **interpretability**. In medicine, understanding *why* a model reached a diagnosis is as crucial as the diagnosis itself.

---

## 🏆 Final Performance Evaluation
Testing on **unseen data** confirmed the model's robustness:

* **General Accuracy:** 97.08%
* **Precision (Malignant):** 98.36%
* **Recall (Malignant):** **93.75%** (Critical metric to minimize False Negatives)
* **F1-Score:** 96.00%



---

## 🔍 Analytical Insights
* **Feature Multicollinearity:** EDA revealed high correlations between size-related metrics (radius, area, perimeter). This was handled through normalization to prevent weight bias.
* **Recall Importance:** The 93.75% recall ensures that the vast majority of malignant cases are caught, which is the primary objective in cancer screening tools.

---

## 🚀 Future Roadmap
* [ ] **Explainable AI (XAI):** Integrate SHAP values to visualize which clinical features most influence a specific diagnosis.
* [ ] **Ensemble Methods:** Test Voting Classifiers to further increase Recall.
* [ ] **External Validation:** Test the model on independent clinical datasets to verify generalization.

---

## ⚙️ Setup & Deployment
1. **Environment:** `python -m venv venv`
2. **Install Dependencies:** `pip install -r requirements.txt`
3. **Run Interactive Dashboard:** `streamlit run app.py`

---
**Giulia Bugatti** 
