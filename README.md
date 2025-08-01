![](UTA-DataScience-Logo.png)

#  Restaurant Inspection Failure Prediction – NYC

## One Line Summary  
Built an interpretable ML system using **Random Forest** and also tested other models to predict restaurant inspection outcomes and help prevent food safety violations.

---

##  Project Overview

###  Objective  
To develop a non-trivial, actionable machine learning model that predicts whether a restaurant is likely to **pass or fail** a health inspection using historical NYC data. This model can assist health departments in prioritizing high-risk inspections and empower businesses to improve compliance. The Random Forest classifier(tuned) stood out as the most reliable and well-rounded model in this project. It achieved the highest performance across all key metrics: an accuracy of 94.88%, an F1 score of 87%, and a ROC AUC of 98%

### Why This Project Matters  
Foodborne illnesses are a major public health concern. When dining out, no one expects to face health issues—or worse, life-threatening consequences. By predicting inspection failures before they occur, this project contributes to:
- Safer food practices  
- Better resource allocation for inspectors  
- Stronger policy enforcement  

---

##  Key Modeling Choices 
- **Target Selection:**  
  I avoided using the `Inspection Score` due to its strong correlation with the final result (risk of label leakage). Instead, I selected `Inspection Result` and converted it into a binary `result` column:  
  - `Pass` = `Satisfactory`, `Complete`  
  - `Fail` = `Unsatisfactory`, `Incomplete`
 
![graph1](graph1.png)

Changed to --->

![graph2](graph2.png)


- **Data Cleaning:**  
  Removed irrelevant or leakage-prone columns like phone number, coordinates, and unique IDs. Also dropped missing values for quality control.

- **Temporal Feature Engineering:**  
  Extracted `year` and `month` from `Inspection Date` to capture trends such as seasonal policy enforcement or cyclical violations.

- **Violation Handling:**  
  Grouped `Violation Description` into the **Top 10 most common** categories plus “Other”, reducing noise from high-cardinality fields.

- **Categorical Encoding:**  
  Applied **one-hot encoding** to fields like `Violation Type` and `Inspection Type` to make them model-compatible without implying any ordering.

- **Class Filtering & Labeling:**  
  Focused on 4 inspection classes and created a new `result` column for binary classification. This made the model’s output easy to interpret for real-world applications.



---

##  Tools

- **Language:** Python  
- **Libraries:** pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, shap  
- **Training Time:** ~10 seconds

---

##  Results Summary: 

The Random Forest classifier emerged as the best-performing model in this project, achieving the highest accuracy (94.88%), F1 score (0.8740), and ROC AUC (0.9825) among all models tested. Its ability to handle complex, high-dimensional data and mitigate overfitting through ensemble learning made it particularly well-suited for predicting restaurant inspection outcomes.

In contrast, while XGBoost is a powerful boosting algorithm, it underperformed in this case with a significantly lower F1 score (0.5486), suggesting poor balance between precision and recall, especially for the minority class. Similarly, Logistic Regression and Decision Tree models offered lower predictive power (F1 scores around 0.51), likely due to their limited capacity to capture non-linear relationships and interactions present in the data. These comparisons highlight why Random Forest was not only the most accurate but also the most robust and generalizable model in this application.

![plot6](plot6.png)

###  Logistic Regression (Primary Baseline)

As a baseline, Logistic Regression performed surprisingly well, given its linear nature. It’s a fast, efficient model that outputs interpretable coefficients, useful for understanding linear feature impacts.

-  Very fast to train and test  
-  Coefficients explain feature impact directionally  
-  Limited in capturing complex patterns or feature interactions  
-  Great for quick prototyping or lightweight apps  
---

###  Decision Tree (Trial model)

The simplest model in our pipeline, Decision Tree offers full transparency in how decisions are made. It’s less accurate than ensemble methods but great for visual explanation.

-  Fully interpretable structure  
-  Quick to train  
-  Prone to overfitting without pruning  
-  Suitable for stakeholder presentations and small-scale rule systems  

###  Random Forest 
Random Forest showed strong overall performance, especially in terms of F1 Score, which is crucial in imbalanced classification problems like this. It’s a robust ensemble model that reduces overfitting by averaging multiple decision trees.

- Robust to overfitting and noise  
- Automatically handles feature interactions
- Tuned using GridSearchCV across parameters like `n_estimators`, `max_depth`, and `min_samples_split`
-  Interpretable via `feature_importances_`  
-  Recommended for use in early deployment phases or when high interpretability is desired  


![plot2](plot2.png)
![plot3](plot3.png)

---

###  XGBoost (Tuned)

XGBoost slightly outperformed all models in **ROC AUC**, indicating it best separates “Pass” from “Fail”. It uses gradient boosting and regularization, making it powerful and efficient.

-  High AUC and precision with hyperparameter tuning
-  Tuned using GridSearchCV for `learning_rate`,` max_depth`, `n_estimators `, etc.
-  Handles missing data and sparse features  
-  Ideal for production systems prioritizing predictive power  
---

![plot5](plot5.png)


###  Final Model Comparison Takeaway

- **XGBoost**: Best for separation and generalization  
- **Random Forest**: Best balance of performance + explainability  
- **Decision Tree**: Best for visual interpretability  
- **Logistic Regression**: Best for speed and simplicity

---
![plot4](plot4.png)


##  Feature Importance Insights

I used both `feature_importances_` from Random Forest and XGBoost to understand what drove the predictions.

###  Top Influential Features:
- Violation Type  
- Grade  
- Inspection Type  
- Inspection Year  
- Violation Description

These features provide **explainability** and can guide inspection training, public health communication, or intervention strategy.

---
##  Real-World Impact

-  Helps inspectors focus on **high-risk businesses**  
-  Allows restaurant owners to **proactively fix** red flags  
-  Can be integrated into a **real-time dashboard** for city officials  
-  Promotes **data-driven food safety policies**

---
##  Future Work

Several enhancements can build on this project to increase its real-world utility:

- **Historical Features:**  
  Include past inspection history (e.g., previous failures, violation trends) for each restaurant to improve prediction accuracy.

- **Geospatial Insights:**  
  Use zip codes or boroughs to uncover regional risk patterns and hotspots.

- **API or Integration:**  
  Build a REST API to allow real-time predictions for internal dashboards or city-wide inspection tools.

##  Reproducibility

##  Run Notebooks in This Order

                                                                                       
 `DataPreprocessing.ipynb`  Clean raw NYC inspection dataset, drop leakage-prone columns, encode labels 
 `LogisticRegression.ipynb` Train and evaluate logistic regression model with ROC/F1 analysis            
 `DecisionTree.ipynb`        Train decision tree classifier, tune `max_depth`, evaluate results          
 `ML_RF.ipynb`               Train and tune Random Forest using `GridSearchCV`, generate visualizations  
 `XGB.ipynb`                 Train and tune XGBoost, tune as well as evaluate with SHAP values and confusion matrix      
 `CompareModels.ipynb`       Generate model comparison table and combined ROC curve for all classifiers  

###  How to Run the Code

1. **Install Dependencies**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap
