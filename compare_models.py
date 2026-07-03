import os
import pandas as pd
import time
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

def load_and_preprocess_data(data_path="app/data/synthetic_medical_triage.csv"):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found at {data_path}")
    
    df = pd.read_csv(data_path)
    
    features = ["age", "heart_rate", "systolic_blood_pressure", "oxygen_saturation", 
                "body_temperature", "pain_level", "chronic_disease_count", "previous_er_visits"]
    X = df[features]
    
    level_mapping = {0: "Routine", 1: "Urgent", 2: "Emergency", 3: "Resuscitation"}
    if df["triage_level"].dtype in ['int64', 'float64']:
        df["triage_level"] = df["triage_level"].map(level_mapping).fillna("Routine")
        
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["triage_level"])
    
    return X, y, label_encoder

def main():
    print("Loading data...")
    X, y, label_encoder = load_and_preprocess_data()
    
    # Split data to ensure fair testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define models
    models = {
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        # Pipeline for scale-sensitive models
        "Logistic Regression": Pipeline([
            ('scaler', StandardScaler()),
            ('logreg', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "K-Nearest Neighbors": Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=5))
        ])
    }
    
    results = []
    
    print("\nTraining and evaluating models...\n")
    print("-" * 80)
    print(f"{'Model Name':<25} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Train Time(s)'}")
    print("-" * 80)
    
    for name, model in models.items():
        # Timing
        start_time = time.time()
        
        # Fit model
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
        
        # Save results
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Training Time (s)": train_time
        })
        
        print(f"{name:<25} | {acc:.4f}     | {precision:.4f}     | {recall:.4f}     | {f1:.4f}     | {train_time:.4f}")
        
    print("-" * 80)
    
    # Convert to DataFrame and sort by F1-Score
    results_df = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False)
    
    # Save report to markdown
    report_path = "model_comparison_report.md"
    with open(report_path, "w") as f:
        f.write("# Model Comparison Report\n\n")
        f.write("This report provides a comparative analysis of different machine learning models evaluated on the hospital triage dataset.\n\n")
        f.write("## Performance Metrics (Test Set - 20%)\n\n")
        f.write(results_df.to_markdown(index=False, float_format="%.4f"))
        f.write("\n\n## Conclusion\n")
        f.write("Based on the evaluation, we can observe the trade-offs between model complexity, accuracy, and training time.\n")
        
    print(f"\nProfessional report saved to {report_path}")

if __name__ == '__main__':
    main()
