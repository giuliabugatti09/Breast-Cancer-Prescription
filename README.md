# 🎗️ Breast Cancer Prediction with Machine Learning

This project develops a machine learning classifier to accurately predict whether a breast tumor is **Malignant (M)** or **Benign (B)** based on features computed from digitized images of fine-needle aspirate (FNA) biopsies.

## 🌟 **Key Features**

✔ **High-Performance Model:** Logistic Regression achieved **97.48% accuracy** in cross-validation after full `GridSearchCV` tuning.

✔ **Robust Model Selection:** Systematic comparison and `GridSearchCV` tuning of 4 top ML algorithms.

✔ **30-Feature Analysis:** Utilizes 30 clinical features (mean, se, worst) for prediction.

✔ **Ready for Deployment:** Includes serialized `.pkl` files for the model (`breast_cancer_model.pkl`) and the scaler (`scaler.pkl`).

## 📊 **Diagnostic Analysis**

The class distribution in the 569-instance dataset is as follows:

  * **Benign (B):** 357 cases
  * **Malignant (M):** 212 cases

## 🔗 **Feature Correlation Matrix**

The exploratory analysis included a heatmap to investigate multicollinearity among the 30 features.

**Key Insight:** The heatmap reveals high positive correlation between features measuring similar characteristics, such as `radius_mean`, `perimeter_mean`, and `area_mean`. This was expected, as they are all metrics of size. Preprocessing with `StandardScaler` ensures the models handle these different scales appropriately.

## 🏆 **Model Performance Comparison**

### Hyperparameter Tuning (GridSearchCV)

Instead of using default parameters, `GridSearchCV` (with 5-fold CV) was used to find the optimal hyperparameters for each of the 4 models. This ensures we are comparing the *best possible version* of each algorithm.

| Algorithm | Best CV Accuracy | Best Parameters |
| :--- | :---: | :--- |
| **Logistic Regression** | **0.9748** | `{'C': 1.0, 'penalty': 'l2'}` |
| KNN | 0.9698 | `{'n_neighbors': 7}` |
| SVM (Linear Kernel) | 0.9672 | `{'C': 0.1}` |
| Random Forest | 0.9522 | `{'max_depth': None, 'min_samples_leaf': 1, 'n_estimators': 50}` |

-----

### Final Model Evaluation: Optimized Logistic Regression

The best model from the `GridSearchCV` (**Logistic Regression**) was then re-trained on the *entire* training set and evaluated on the *unseen* **test set**.

| Metric | Test Set Score |
| :--- | :---: |
| **Accuracy** | 0.9708 |
| **Precision** (Malignant) | 0.9836 |
| **Recall** (Malignant) | 0.9375 |
| **F1-Score** (Malignant) | 0.9600 |

### Recommendation

**Logistic Regression** is the recommended model. It not only performed best in the rigorous cross-validation tuning (**97.48%**) but also proved excellent generalization on the test set (**97.08%** accuracy).

The **Recall of 0.9375** is a crucial metric in this medical context, indicating the model correctly identified nearly 94% of all malignant cases, thereby minimizing false negatives.

## 🛠️ **Technical Implementation**

### **Data Pipeline**

```
1. Load Data → 2. EDA → 3. Clean (Drop 'id', 'Unnamed: 32') → 4. Encode (Diagnosis) → 5. StandardScaler → 6. Split (70/30) → 7. GridSearchCV Tuning → 8. Final Evaluation
```

## 📂 **Project Structure**

```
breast-cancer-prediction/
├── images/                 # Folder for storing images (plots, matrices)
├── data/                   # Folder for storing the dataset
├── app.py                  # Web application script (Streamlit)
├── notebook.ipynb          # Jupyter Notebook with exploratory analysis and modeling
├── requirements.txt        # List of Python dependencies for the project
├── README.md               # Project documentation
```

## 🔮 **Future Enhancements**

  * [ ] Use SHAP or LIME to explain model predictions.
  * [ ] Test on other breast cancer datasets to validate generalization.

## ✉️ **Contact**

**Giulia Bugatti**
[](https://www.google.com/search?q=%5Bhttps://linkedin.com/in/giulia-bugatti-fonseca-226955267/%5D\(https://linkedin.com/in/giulia-bugatti-fonseca-226955267/\))
[](mailto:giuliabugatti02@gmail.com)

-----

**Dataset Reference:**
Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [[http://archive.ics.uci.edu/ml](http://archive.ics.uci.edu/ml)]. Irvine, CA: University of California, School of Information and Computer Science.
