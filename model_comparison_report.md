# Model Comparison Report

This report provides a comparative analysis of different machine learning models evaluated on the hospital triage dataset.

## Performance Metrics (Test Set - 20%)

| Model Name          | Accuracy | Precision | Recall | F1-Score | Train Time (s) |
|---------------------|----------|-----------|--------|----------|----------------|
| **Random Forest**   | 0.9381   | 0.9375    | 0.9381 | 0.9375   | 19.06          |
| **Logistic Regression**| 0.9375   | 0.9369    | 0.9375 | 0.9370   | 4.20           |
| **XGBoost**         | 0.9364   | 0.9358    | 0.9364 | 0.9359   | 15.46          |
| **K-Nearest Neighbors**| 0.9208 | 0.9199    | 0.9208 | 0.9200   | 1.10           |

> [!NOTE]
> The metric values above are calculated on a holdout test set (20% of the dataset) using a `stratify` split to ensure fair class distribution.

## Key Insights

1. **Random Forest vs. XGBoost**: The Random Forest model very slightly outperformed XGBoost on this dataset (0.9375 F1-Score vs 0.9359), though it took slightly longer to train (19s vs 15.4s). This suggests that for this specific dataset with default hyperparameters, bagging might be marginally better than boosting, or XGBoost might require further hyperparameter tuning to surpass Random Forest.
2. **Logistic Regression Surprise**: Logistic Regression performed incredibly well (0.9370 F1-Score), practically matching the complex tree-based models while taking significantly less time to train (4.2 seconds). This indicates that the relationship between the features and the triage levels is highly linear or that the dataset might not have complex non-linear boundaries.
3. **Training Time Trade-offs**: K-Nearest Neighbors was the fastest to train (1.10s) but had the lowest overall performance (0.9200 F1-Score). 

