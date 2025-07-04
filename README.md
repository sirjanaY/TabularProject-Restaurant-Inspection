![UTA-DataScience-Logo](UTA-DataScience-Logo.png)

# XGBoost for Restaurant Inspection Prediction

* **One Sentence Summary**: This repository contains a solution using XGBoost to classify restaurant inspection results as Pass or Fail, based on data from the City of Austin Health Inspection Records.

## Overview

* **Definition of the tasks / challenge**: The task is to predict the outcome (Pass/Fail) of a restaurant inspection based on features from a dataset of previous inspection records.
* **Our approach**: We approached the problem as a binary classification task using the XGBoost algorithm. The data was cleaned, categorical features were encoded, and missing values were handled. We experimented with hyperparameter tuning and model evaluation to get the best possible accuracy.
* **Summary of the performance achieved**: Our best XGBoost model achieved an accuracy of approximately 91% with a high ROC AUC score and precision for both classes.

## Summary of Work Done

### Data

* **Type**:
  * Input: CSV file with features like risk category, inspection score, inspection type, and violation codes.
  * Output: Inspection result (Pass/Fail).
* **Size**: ~88,000 records.
* **Split**: 80% training, 20% testing.

#### Preprocessing / Cleanup
* Removed irrelevant columns like 'Name' during model training and added it back for submission.
* Encoded categorical variables.
* Imputed missing values.

#### Data Visualization
* Bar plots for Pass/Fail distribution.
* Count plots for categorical features.

### Problem Formulation

* **Input / Output**:
  * Input: Preprocessed inspection features
  * Output: Binary label ('Pass'=0, 'Fail'=1)
* **Models**:
  * Random Forest (baseline)
  * XGBoost (final model)
* **Loss / Optimizer**:
  * Loss: Log-loss
  * Hyperparameters: learning rate = 0.1, max depth = 3, n_estimators = 100

### Training
* **Software/Hardware**: Python (pandas, sklearn, xgboost), ran on local machine with 8GB RAM.
* **Time**: ~10 seconds to train.
* **Stopping Criteria**: Based on test performance.
* **Difficulties**: Handling high cardinality features and overfitting. Used subsampling and column sampling to mitigate it.

### Performance Comparison

| Model          | Accuracy | Precision | Recall | ROC AUC |
|----------------|----------|-----------|--------|---------|
| Random Forest  | ~89%     | 0.88      | 0.85   | 0.90    |
| **XGBoost**     | **91%**  | 0.91      | 0.87   | 0.92    |

* **Visualization**:
  * Confusion matrix
  * ROC Curve

### Conclusions

* XGBoost consistently outperformed Random Forest in both accuracy and AUC.
* Careful feature engineering and class balancing helped boost performance.

### Future Work

* Try ensemble stacking with logistic regression or neural networks.
* Incorporate time-based features for more insight.
* Deploy model for real-time inspection prediction.

## How to Reproduce Results

* Run `Prototype.solution.ipynb` to train the model.
* Run `Feasibility.Solution.ipynb` to generate and export the submission CSV.
* Make sure `cleaned_inspection_data.csv` is available.

### Overview of Files in Repository

* `Prototype.solution.ipynb`: Initial data processing and Random Forest baseline.
* `Feasibility.Solution.ipynb`: Final XGBoost model, evaluation, and submission creation.
* `submission_xgboost.csv`: Final predictions for submission.
* `cleaned_inspection_data.csv`: Cleaned dataset used for modeling.

### Software Setup

* Required packages:
  * pandas
  * numpy
  * matplotlib
  * seaborn
  * xgboost
  * scikit-learn

```bash
pip install pandas numpy matplotlib seaborn xgboost scikit-learn
```

### Data

* Download original dataset from the City of Austin Health Inspection Records.
* Use preprocessing script in the notebook to clean and format the data.

### Training

* Run cell blocks in the notebook to train and evaluate the model.

#### Performance Evaluation

* Evaluate with confusion matrix, classification report, and ROC curve in notebook.

## Citations

* XGBoost Documentation: https://xgboost.readthedocs.io/
* Scikit-learn Documentation: https://scikit-learn.org/
* Kaggle Dataset (if applicable): [Insert Kaggle link if used]
