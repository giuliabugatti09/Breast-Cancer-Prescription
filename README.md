# Breast-Cancer-Prescription

Perfeito! Abaixo está um modelo de documentação `README.md` **organizado, profissional e atrativo** para seu projeto de **Breast Cancer Prediction**, pronto para ser usado no GitHub:

---

## 🩺 Breast Cancer Prediction

This project aims to build a machine learning model to accurately classify breast cancer diagnoses as **malignant (M)** or **benign (B)** using data from digitized images of fine needle aspirates (FNA) of breast masses.

### 📁 Dataset

* **Source**: [Breast Cancer Wisconsin (Diagnostic) Data Set – UCI Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
* **Features**: 30 numeric features computed from cell nuclei images.
* **Target**: Diagnosis (`M` for malignant, `B` for benign).

---

### 🧠 Objective

To apply supervised learning techniques for early and accurate breast cancer prediction, assisting healthcare decisions.

---

### 🛠️ Project Structure

```bash
📂 breast-cancer-prediction/
│
├── 📄 README.md              # Project documentation
├── 📊 cancerPrediction.ipynb  # Main notebook
├── 📁 data/                  # (Optional) Folder for raw/processed datasets
```

---

### 🔎 Main Steps

1. **📥 Data Loading**
   Read the CSV dataset and inspect its structure.

2. **📊 Exploratory Data Analysis (EDA)**
   Understand feature distributions, correlations and class balance.

3. **🧹 Data Preprocessing**

   * Handle missing values
   * Encode target labels
   * Scale features with `StandardScaler`
   * Train-test split

4. **🤖 Model Training**
   Trained multiple models:

   * Logistic Regression
   * Support Vector Machine (SVM)
   * K-Nearest Neighbors (KNN)
   * Random Forest

5. **📈 Evaluation**
   Metrics used:

   * Accuracy, Precision, Recall, F1-Score
   * Confusion Matrix
   * Cross-Validation (CV) Scores

---

### 🏆 Best Results

| Model               | CV Accuracy           |
| ------------------- | --------------------- |
| Logistic Regression | 0.9978 ± 0.0044       |
| SVM (Linear)        | **1.0000 ± 0.0000** ✅ |
| KNN (k=5)           | 0.9956 ± 0.0088       |
| Random Forest       | 0.9978 ± 0.0044       |

> The SVM model achieved perfect performance in cross-validation.

---

### ✅ Conclusion

This model shows strong potential as a diagnostic support tool for breast cancer detection. With high accuracy and precision, it can assist professionals in making timely and accurate medical decisions.

---

### 🚀 Future Improvements

* Deploy via Streamlit or Flask for user interface.
* Test on real-world datasets for robustness.
* Improve interpretability using SHAP or LIME.

---

### 📌 Requirements

```bash
scikit-learn
pandas
numpy
matplotlib
seaborn
```

> Install with: `pip install -r requirements.txt` (optional: create one)

---

### 🙋‍♀️ Author

* **Giulia Bugatti**
  🔗 [Portfolio](https://giuliabugatti09.github.io) | 🐙 [GitHub](https://github.com/giuliabugatti09)

---

Se quiser, posso gerar o arquivo `README.md` pra você, montar um botão de download ou adicionar badges (como stars, license, etc). Deseja isso?
