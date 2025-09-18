# 🩺 Heart Stroke Prediction using Machine Learning

This repository implements the research paper:  
**"Stacking Model for Heart Stroke Prediction using Machine Learning Techniques"**  
by Subasish Mohapatra et al., *EAI Endorsed Transactions on Pervasive Health and Technology*, 2023.  

The project uses **machine learning models** and a **stacking ensemble** to predict the risk of stroke in patients based on medical attributes.  
It also provides an **interactive Streamlit dashboard** with **SHAP & LIME explanations** for interpretability.


---

## 📖 Overview
Cardiovascular diseases (CVDs) and stroke are leading causes of death worldwide.  
Early prediction of stroke risk can help in **timely intervention and treatment**.

This project:
- Trains multiple **ML algorithms** (Logistic Regression, Random Forest, Gradient Boosting, etc.)
- Uses a **Stacking Ensemble model** to improve accuracy
- Achieves up to **97.67% accuracy** (Gradient Boosting)
- Deploys a **Streamlit web app** for real-time predictions
- Adds **Explainability** with **SHAP & LIME** to show which factors contributed most to predictions

---

## 📊 Dataset
We used the **[Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)** from Kaggle.

- **Rows**: ~5,000 patient records  
- **Features**:
  - Gender
  - Age
  - Hypertension
  - Heart Disease
  - Ever Married
  - Work Type
  - Residence Type
  - Average Glucose Level
  - BMI
  - Smoking Status
- **Target**: `stroke` (0 = No stroke, 1 = Stroke)

---

## 🛠 Methodology
1. **Data Preprocessing**
   - Handling missing values (`BMI`)
   - Encoding categorical features
   - Scaling numerical features
   - Train-test split (80:20)

2. **Model Training**
   - Algorithms: LR, KNN, SVM, Naïve Bayes, Decision Tree, Random Forest, Gradient Boosting, AdaBoost, Extra Trees
   - 10-fold cross-validation
   - Metrics: Accuracy, Precision, Recall, F1-score

3. **Ensemble Learning**
   - Implemented **Stacking Classifier**
   - Final estimator: Gradient Boosting

4. **Evaluation**
   - Gradient Boosting achieved **94.67% accuracy**
   - Metrics compared across models

5. **Deployment**
   - Streamlit dashboard for predictions
   - Added SHAP & LIME explanations

---

## 🌟 Features
- Real-time **stroke risk prediction**
- **Probability score** for predictions
- **Interactive dashboard** with input sliders
- **Explainability with SHAP & LIME**
- Model performance metrics
- Confusion matrix & feature importance plots

---

## 💻 Tech Stack
- **Language**: Python 3.10+
- **Libraries**:
  - `scikit-learn`
  - `pandas`, `numpy`
  - `matplotlib`, `seaborn`
  - `joblib`
  - `streamlit`
---

## ⚙️ Installation

Clone the repository:
```bash
git clone https://github.com/your-username/stroke-prediction.git
cd stroke-prediction
pip install -r requirements.txt
```
Run Streamlit Dashboard:
```bash
streamlit run app.py