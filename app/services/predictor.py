import os
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.preprocessing import LabelEncoder

class TriagePredictor:
    def __init__(self, model_path="app/data/triage_model.json", data_path="app/data/synthetic_medical_triage.csv"):
        self.model_path = model_path
        self.data_path = data_path
        self.encoder_path = "app/data/label_encoder.joblib"
        self.model = None
        self.label_encoder = LabelEncoder()
        
        # Initialize and load or train the model
        self.initialize_model()

    def load_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Training data not found at {self.data_path}")
        return pd.read_csv(self.data_path)

    def initialize_model(self):
        """Loads the model if it exists, otherwise trains a new one."""
        os.makedirs("app/data", exist_ok=True)
        
        if os.path.exists(self.model_path) and os.path.exists(self.encoder_path):
            self.model = xgb.XGBClassifier()
            self.model.load_model(self.model_path)
            self.label_encoder = joblib.load(self.encoder_path)
        else:
            self.train_and_save_model()

    def train_and_save_model(self):
        """Trains the XGBoost model using the synthetic triage dataset."""
        df = self.load_data()
        
        features = ["age", "heart_rate", "systolic_blood_pressure", "oxygen_saturation", 
                    "body_temperature", "pain_level", "chronic_disease_count", "previous_er_visits"]
        X = df[features]
        
        # Label decoding dictionary to ensure readable names if numeric
        # Assume 0: Routine, 1: Urgent, 2: Emergency, 3: Resuscitation
        level_mapping = {0: "Routine", 1: "Urgent", 2: "Emergency", 3: "Resuscitation"}
        if df["triage_level"].dtype in ['int64', 'float64']:
            df["triage_level"] = df["triage_level"].map(level_mapping).fillna("Routine")
        
        # Encode the target labels
        y = self.label_encoder.fit_transform(df["triage_level"])
        
        self.model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
        self.model.fit(X, y)
        
        # Save model and encoder
        self.model.save_model(self.model_path)
        joblib.dump(self.label_encoder, self.encoder_path)

    def predict(self, input_data: dict) -> str:
        """Predicts the triage level for incoming patient data."""
        features = ["age", "heart_rate", "systolic_blood_pressure", "oxygen_saturation", 
                    "body_temperature", "pain_level", "chronic_disease_count", "previous_er_visits"]
        
        # Convert input dictionary to a DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Ensure only the necessary features are passed in the correct order
        X_input = input_df[features]
        
        # Make prediction
        prediction = self.model.predict(X_input)
        
        # Decode the numerical prediction back to a string label (e.g., "Emergency", "Urgent")
        predicted_label = self.label_encoder.inverse_transform(prediction)[0]
        
        return predicted_label

# Instantiate a global predictor object for use in endpoints
predictor_service = TriagePredictor()