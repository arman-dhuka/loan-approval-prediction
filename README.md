# 🏦 Loan Approval Prediction System

An end-to-end Machine Learning project that predicts whether a loan application will be **Approved** or **Rejected** based on applicant details. Built with a complete ML pipeline — from data cleaning to deployment — featuring a real-time **Streamlit** web interface with confidence scores.

<!-- ![Loan Approval Prediction Demo](assets/demo_screenshot.png) -->

---

## 🔗 Live Demo

> 🚀 https://loan-approval-prediction-by-arman.streamlit.app/
---

## 📌 Project Overview

The **Loan Approval Prediction System** is designed to assist financial institutions in automating the loan eligibility process. By analyzing key applicant attributes — income, credit history, education, and more — the model predicts loan approval with **80%+ accuracy**.

### Why This Project?

- Solves a **real-world business problem** in the banking/finance domain
- Demonstrates a **production-ready ML workflow** from raw data to deployment
- Showcases skills in **data analysis, feature engineering, model tuning, and deployment**

---

## ✨ Features

- ✅ End-to-end ML pipeline (EDA → Preprocessing → Feature Engineering → Training → Tuning → Deployment)
- ✅ Multiple ML models compared and evaluated
- ✅ Custom feature engineering for better model performance
- ✅ Hyperparameter tuning using RandomizedSearchCV / GridSearchCV
- ✅ Interactive **Streamlit** web app with two-column responsive layout
- ✅ Real-time loan prediction with **probability/confidence score**
- ✅ Clean, modular, production-level Python codebase

---

## 📊 Dataset

**Source:** [Loan Prediction Dataset — Analytics Vidhya / Kaggle](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)

| Feature            | Description                              | Type        |
|--------------------|------------------------------------------|-------------|
| Gender             | Male / Female                            | Categorical |
| Married            | Applicant married (Yes / No)             | Categorical |
| Dependents         | Number of dependents (0, 1, 2, 3+)       | Categorical |
| Education          | Graduate / Not Graduate                  | Categorical |
| Self_Employed      | Self-employed (Yes / No)                 | Categorical |
| ApplicantIncome    | Applicant's monthly income               | Numerical   |
| CoapplicantIncome  | Co-applicant's monthly income            | Numerical   |
| LoanAmount         | Loan amount (in thousands)               | Numerical   |
| Loan_Amount_Term   | Term of loan (in months)                 | Numerical   |
| Credit_History     | Credit history meets guidelines (1 / 0)  | Categorical |
| Property_Area      | Urban / Semiurban / Rural                | Categorical |
| Loan_Status        | Loan approved (Y / N) — **Target**       | Binary      |

**Dataset Size:** 614 records with 12 features

---

## 🔄 ML Workflow

The project follows a structured, step-by-step machine learning pipeline:

```
📥 Raw Data
   │
   ▼
📊 Exploratory Data Analysis (EDA)
   │  → Distribution analysis, correlation heatmaps, outlier detection
   ▼
🧹 Data Preprocessing
   │  → Handle missing values (mode/median imputation)
   │  → Drop Loan_ID, fix data types
   │  → Encode categorical variables (Label + One-Hot Encoding)
   ▼
🔧 Feature Engineering
   │  → Total_Income = ApplicantIncome + CoapplicantIncome
   │  → Income_Loan_Ratio = Total_Income / LoanAmount
   ▼
🤖 Model Training
   │  → Train-test split (80/20)
   │  → Train multiple classifiers
   │  → Evaluate with accuracy, precision, recall, F1
   ▼
🎯 Hyperparameter Tuning
   │  → RandomizedSearchCV / GridSearchCV
   │  → Optimize best-performing model
   ▼
🚀 Deployment
   └  → Streamlit web application with real-time predictions
```

---

## 🔧 Feature Engineering

Two custom features were engineered to improve model performance:

| New Feature         | Formula                                     | Rationale                                                |
|---------------------|---------------------------------------------|----------------------------------------------------------|
| `Total_Income`      | `ApplicantIncome + CoapplicantIncome`       | Combined household income is a stronger predictor         |
| `Income_Loan_Ratio` | `Total_Income / LoanAmount`                 | Measures repayment capacity relative to loan size         |

```python
# src/feature_engineering.py
df["Total_Income"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
df["Income_Loan_Ratio"] = df["Total_Income"] / df["LoanAmount"]
```

These features capture the **repayment ability** of the applicant more effectively than raw income alone.

---

## 🤖 Models Used

| #  | Model                  | Description                                          |
|----|------------------------|------------------------------------------------------|
| 1  | Logistic Regression    | Baseline model — fast, interpretable                 |
| 2  | Decision Tree          | Non-linear classifier — captures complex patterns    |
| 3  | Random Forest          | Ensemble of decision trees — reduces overfitting     |
| 4  | **XGBoost** ⭐          | Gradient boosting — best performance after tuning    |

---

## 📈 Model Performance

### Before Improvement (Baseline)

| Model               | Accuracy  |
|----------------------|-----------|
| Logistic Regression  | ~78%      |
| Decision Tree        | ~68%      |
| Random Forest        | ~77%      |

### After Feature Engineering + Hyperparameter Tuning

| Model                     | Accuracy  |
|---------------------------|-----------|
| Logistic Regression       | ~79%      |
| Random Forest (Tuned)     | ~80%      |
| **XGBoost (Tuned)** ⭐     | **~81%+** |

> 📌 The tuned XGBoost model was selected as the **final production model** due to its superior generalization performance.

---

## 🎯 Hyperparameter Tuning

The final model was optimized using **RandomizedSearchCV** and **GridSearchCV** with the following approach:

```python
# Example: XGBoost Hyperparameter Space
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
}
```

**Strategy:**
1. Used `RandomizedSearchCV` for initial broad search across hyperparameter space
2. Followed up with `GridSearchCV` for fine-grained tuning around best parameters
3. Evaluated using **5-fold cross-validation** to ensure robust performance
4. Saved the best-performing model as `tuned_model.pkl`

---

## 📁 Project Structure

```
loan-approval-prediction/
│
├── 📂 app/
│   └── app.py                        # Streamlit web application
│
├── 📂 data/
│   ├── raw/
│   │   └── loan_data.csv             # Original dataset
│   └── processed/
│       ├── clean_data.csv            # After preprocessing
│       └── featured_data.csv         # After feature engineering
│
├── 📂 notebooks/
│   ├── eda.ipynb                     # Exploratory Data Analysis
│   ├── preprocessing.ipynb           # Data cleaning & encoding
│   ├── feature_engineering.ipynb     # Custom feature creation
│   ├── model_training.ipynb          # Model training & evaluation
│   └── hyperparameter_tuning.ipynb   # Tuning with Grid/Random Search
│
├── 📂 src/
│   ├── data_preprocessing.py         # Preprocessing pipeline script
│   ├── feature_engineering.py        # Feature engineering script
│   └── train_model.py                # Model training script
│
├── 📂 models/
│   ├── loan_model.pkl                # Baseline trained model
│   └── tuned_model.pkl               # Tuned production model
│
├── 📂 config/                         # Configuration files
├── requirements.txt                   # Python dependencies
├── runtime.txt                        # Python version (3.11)
└── README.md                          # Project documentation
```

---

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.11+
- pip (Python package manager)
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/arman-dhuka/loan-approval-prediction.git
cd loan-approval-prediction
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit App

```bash
streamlit run app/app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## 🖥️ Example Output

### ✅ Loan Approved

```
🔍 Prediction Result
──────────────────────
✅ Loan Approved (Confidence: 0.87)
```

### ❌ Loan Rejected

```
🔍 Prediction Result
──────────────────────
❌ Loan Rejected (Confidence: 0.73)
```

> The app displays a **confidence score** with each prediction, helping users understand the model's certainty.

---

## 🔮 Future Improvements

- [ ] Deploy on **Streamlit Community Cloud** for public access
- [ ] Add **SHAP / LIME** explainability for individual predictions
- [ ] Implement **deep learning models** (Neural Networks) for comparison
- [ ] Add **user authentication** and prediction history tracking
- [ ] Integrate a **REST API** (FastAPI) for production-grade serving
- [ ] Build a **CI/CD pipeline** for automated model retraining
- [ ] Add **more feature engineering** (e.g., EMI calculation, loan-to-income bins)

---

## 📝 Resume-Ready Description

> **Loan Approval Prediction System** — Built an end-to-end ML pipeline to predict loan approval using Logistic Regression, Decision Tree, Random Forest, and XGBoost. Performed EDA, data preprocessing, custom feature engineering (Total_Income, Income_Loan_Ratio), and hyperparameter tuning (RandomizedSearchCV/GridSearchCV), achieving **81%+ accuracy**. Deployed a real-time prediction interface using **Streamlit** with confidence scoring. Tech stack: Python, Scikit-learn, XGBoost, Pandas, Streamlit.

---

## 🧰 Tech Stack

| Technology     | Usage                          |
|----------------|--------------------------------|
| Python 3.11    | Core programming language      |
| Pandas         | Data manipulation & analysis   |
| NumPy          | Numerical computing            |
| Scikit-learn   | ML models, pipelines, tuning   |
| XGBoost        | Gradient boosting classifier   |
| Streamlit      | Web app deployment             |
| Joblib         | Model serialization            |
| Matplotlib     | Data visualization             |
| Seaborn        | Statistical visualization      |

---

## 👨‍💻 Author

**Arman Dhuka**

- 🔗 GitHub: [@arman-dhuka](https://github.com/arman-dhuka)
<!-- - 🔗 LinkedIn: [Arman Dhuka](https://linkedin.com/in/arman-dhuka) -->

---

## ⭐ Show Your Support

If you found this project helpful, give it a ⭐ on GitHub — it helps others discover it too!

---

<p align="center">Made with ❤️ by Arman Dhuka</p>