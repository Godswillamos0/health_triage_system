from pydantic import BaseModel, Field

class TriageRequest(BaseModel):
    age: float = Field(..., description="Patient age in years")
    heart_rate: float = Field(..., description="Heart rate in bpm")
    systolic_blood_pressure: float = Field(..., description="Systolic blood pressure")
    oxygen_saturation: float = Field(..., description="Oxygen saturation percentage")
    body_temperature: float = Field(..., description="Body temperature in °C")
    pain_level: int = Field(..., description="Pain level (0-10)")
    chronic_disease_count: int = Field(..., description="Number of chronic diseases")
    previous_er_visits: int = Field(..., description="Number of previous ER visits")
    symptoms: str = Field(..., description="Chief complaint or symptoms")
    latitude: float = Field(7.2507, description="Patient latitude")
    longitude: float = Field(5.2069, description="Patient longitude")

class TriageResponse(BaseModel):
    triage_level: str
    urgency_score: int
    reasoning: str
    recommended_facilities: list