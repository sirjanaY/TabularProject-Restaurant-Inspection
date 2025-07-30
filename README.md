![UTA-DataScience-Logo](UTA-DataScience-Logo.png)

#  Restaurant Inspection Failure Prediction – NYC

**One Liner**: Built an interpretable ML system using XGBoost & Random Forest to predict restaurant inspection outcomes and help prevent food safety violations.

---

##  Project Overview

###  Objective
To develop a **non-trivial**, actionable machine learning model that predicts if a restaurant is likely to **pass or fail** a health inspection using historical NYC data. This model can assist health departments in **prioritizing high-risk inspections** and empower businesses to improve compliance.

###  Why This Project Matters
Foodborne illnesses are a major public health concern. By predicting inspection failures before they happen, this project contributes to **safer food practices**, **resource allocation**, and **better policy enforcement**.

---

##  Key Modeling Choices & Justifications

| Step | What We Did | Why It Matters |
|------|-------------|----------------|
|  **Target Selection** | Used **`Inspection Result`** instead of `Inspection Score` | To avoid label leakage — `Score` is often directly correlated with outcome, making it **trivial**. `Result` is a qualitative label, requiring the model to **understand deeper patterns**. |
|  **Data Cleaning** | Removed identifiers (e.g. phone, coordinates), dropped missing values | These columns are either **irrelevant or leakage-prone** and can bias the model |
|  **Date Features** | Extracted **year** and **month** from inspection date | Time-based trends often affect inspection outcomes — seasonality, policy changes, etc. |
|  **Violation Handling** | Grouped `Violation Description` into Top 10 categories + "Other" | Avoids high-cardinality issues and **focuses on most frequent violations** |
|  **Categorical Encoding** | One-hot encoded key fields like `Violation Type`, `Inspection Type` | Machine learning models need numerical inputs; one-hot encoding prevents **ordinal misinterpretation** |
|  **Class Filtering** | Filtered dataset to only 4 interpretable classes (`Satisfactory`, `Complete`, `Unsatisfactory`, `Incomplete`) | Ensures **label consistency**, removes ambiguous or rarely used labels |
|  **Model Comparison** | Compared **Random Forest** (baseline) vs. **XGBoost** (optimized) | To evaluate trade-offs between interpretability, performance, and class separation |

---

## Tech Stack

- **Language**: Python  
- **Libraries**: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `shap`  
- **Hardware**: Local machine with 8GB RAM  
- **Training Time**: ~10 seconds

---

## 📈 Results Summary

| Metric            | Random Forest | XGBoost       |
|------------------|---------------|---------------|
| **Accuracy**      | **94.88%**    | 94.55%        |
| **F1 Score**      | **87.44%**    | 85.98%        |
| **ROC AUC Score** | 98.25%        | **98.29%**    |

- 🔎 **RF** had stronger overall classification performance  
- 🔬 **XGBoost** excelled in separating classes (ROC AUC)

---

## 🔥 Feature Importance Insights

Used both `.feature_importances_` from RF and `SHAP` for model explainability:

- Key Drivers:  
  - `Violation Type`
  - `Grade`
  - `Inspection Type`
  - `Inspection Year`
  - `Violation Description`

These features help **explain what drives failure** and could guide training or resource allocation.

---

##  Real-World Impact

-  Helps inspectors **focus on at-risk businesses**
-  Allows restaurant owners to **predict and fix red flags** before failing
- Can be integrated into a **real-time dashboard** for the Department of Health

---

## Reproducibility

###  How to Run:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap
