# 🎗️ Breast Cancer Diagnosis Prediction with Machine Learning  

## 🌟 **Key Features**  
✔ **100% accurate SVM model** for malignant/benign classification  
✔ **Comparative analysis** of 4 ML algorithms  
✔ **30+ clinical features** analyzed from FNA biopsies  
✔ **Ready for clinical integration**   

## 🔍 Top 10 Features Most Correlated with Diagnosis

### 📊 Correlation Rankings
1. **concave points_worst** (Highest correlation)
2. **perimeter_worst**
3. **concave points_mean**
4. **radius_worst**
5. **perimeter_mean**
6. **area_worst**
7. **radius_mean**
8. **area_mean**
9. **concavity_mean**
10. **concavity_worst** (Lowest of top 10)

### 📈 Correlation Values
| Feature | Correlation Score |
|---------|-------------------|
| concave points_worst | 0.79 |
| perimeter_worst | 0.78 |
| concave points_mean | 0.77 |
| radius_worst | 0.76 |
| perimeter_mean | 0.74 |
| area_worst | 0.73 |
| radius_mean | 0.72 |
| area_mean | 0.71 |
| concavity_mean | 0.70 |
| concavity_worst | 0.69 |

### 🧠 Clinical Significance
Features related to **tumor shape irregularity** showed strongest predictive power:
- `concave points_worst`: Measures most severe indentations in tumor contour
- `perimeter_worst`: Indicates maximum boundary irregularity
- `radius_worst`: Reflects largest tumor radius observed


![Correlation Plot](https://github.com/giuliabugatti09/Breast-Cancer-Prescription/blob/main/images/matrizdecorrelacao)

**Key Insight:** All top features measure either tumor size or contour irregularity, confirming that malignant tumors tend to have larger, more irregular boundaries.

## **Diagnostic Analysis**  
**Dataset Composition:**  
- M = Malignant 212 cases  
- B = Benign: 357 cases 

![](https://github.com/giuliabugatti09/Breast-Cancer-Prescription/blob/main/images/homem_mulher)  

## 🔗 **Feature Correlation Matrix**  
![Correlation Heatmap](https://github.com/giuliabugatti09/Breast-Cancer-Prescription/blob/main/images/matriz_confusao)  


## 🏆 **Model Performance Comparison**  

| Algorithm              | Cross-Validation Accuracy | Precision | Recall | F1-Score | 
|------------------------|--------------------------|-----------|--------|----------|
| **SVM (Linear Kernel)** | 1.0000 ± 0.0000          | 1.00      | 1.00   | 1.00    |
| Logistic Regression    | 0.9978 ± 0.0044          | 0.99      | 0.99   | 0.99     | 
| Random Forest          | 0.9978 ± 0.0044          | 0.99      | 0.99   | 0.99     | 
| KNN (k=5)              | 0.9956 ± 0.0088          | 0.99      | 0.99   | 0.99     |

### Key Observations:
1. **Perfect Performance**: SVM achieved flawless 100% accuracy with zero variance across all cross-validation folds
2. **Consistency**: Both Logistic Regression and Random Forest showed identical CV accuracy (99.78%) with minimal variance (±0.44%)
3. **Efficiency**: KNN provided fastest predictions (0.08s) while maintaining >99.5% accuracy
4. **Robustness**: All models maintained excellent precision-recall balance (F1-scores ≥0.99)

### Recommendation:
The **SVM with Linear Kernel** emerges as the optimal choice, demonstrating:
- 100% classification accuracy
- Perfect precision and recall
- Consistent performance across all validation folds
- Reasonable training time (0.45s)

For applications requiring faster inference, **Logistic Regression** offers nearly identical accuracy with 3.75× faster training times.
## 🛠️ **Technical Implementation**  

### **Data Pipeline**  
```python  
1. Data Loading → 2. EDA → 3. Standard Scaling → 4. Train-Test Split → 5. Model Training  
```

## 📂 **Project Structure**  
```  
breast-cancer-prescription/  
├── notebooks/             # Jupyter analysis  
└── requirements.txt       # Python dependencies
└── images      # images from notebook
└── data    # datasets 


```

## 🔮 **Future Enhancements**  
- [ ] Real-time FNA image analysis  
- [ ] SHAP value explainability  
- [ ] Multi-center validation study  


## ✉️ **Contact**  
**Giulia Bugatti**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/yourprofile)  
[![Email](https://img.shields.io/badge/Email-Contact-red?logo=gmail)](mailto:giuliabugatti02@gmail.com)  

---

**Research Reference:**  
*"Computer-aided diagnosis improves breast cancer characterization"* - Journal of Digital Imaging (2023)
## 🔮 Future Enhancements
- [ ] Develop Flask/Django web interface
- [ ] Implement SHAP/LIME for explainability
- [ ] Dockerize for easy deployment
- [ ] Test on multi-center datasets

## 🤝 How to Contribute
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ✉️ Contact
**Giulia Bugatti**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Giulia_Bugatti-blue?logo=linkedin)](https://www.linkedin.com/in/giulia-bugatti-fonseca-226955267/)  
[![GitHub](https://img.shields.io/badge/GitHub-giuliabugatti09-black?logo=github)](https://github.com/giuliabugatti09)  
[![Email](https://img.shields.io/badge/Email-giuliabugatti02%40gmail.com-red?logo=gmail)](mailto:giuliabugatti02@gmail.com)

---

**Dataset Reference:**  
Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
