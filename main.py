from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from app.schemas.triage import TriageRequest, TriageResponse
from app.services.nlp_engine import NLPEngine
from app.services.predictor import predictor_service
from app.services.recommender import recommender_service
import uvicorn

app = FastAPI(title="Intelligent Decision Support System for Triage")

templates = Jinja2Templates(directory="app/templates")
nlp_engine = NLPEngine()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/api/triage", response_model=TriageResponse)
async def triage(request: TriageRequest):
    # NLP Engine
    extracted_symptoms = nlp_engine.extract_symptoms(request.symptoms)
    
    # ML Predictor
    input_data = request.model_dump(exclude={"symptoms", "latitude", "longitude"})
    triage_level = predictor_service.predict(input_data)
    
    # Recommender
    symptoms_text = request.symptoms.lower()
    
    # Simple rule-based specialty selection
    if triage_level in ["Emergency", "Resuscitation"]:
        specialty = "Emergency"
    elif "maternity" in symptoms_text or "pregnant" in symptoms_text or "labor" in symptoms_text:
        specialty = "Maternity"
    elif "fracture" in symptoms_text or "bone" in symptoms_text or "break" in symptoms_text:
        specialty = "Orthopedics"
    elif "child" in symptoms_text or "baby" in symptoms_text or request.age < 18:
        specialty = "Pediatrics"
    elif "surgery" in symptoms_text:
        specialty = "Surgery"
    else:
        specialty = "General"
        
    recommended_facilities = recommender_service.rank_facilities(
        user_lat=request.latitude,
        user_lon=request.longitude,
        required_specialty=specialty,
        limit=3
    )
    
    urgency_scores = {"Routine": 1, "Urgent": 2, "Emergency": 3, "Resuscitation": 4}
    urgency_score = urgency_scores.get(triage_level, 1)
    
    reasoning = f"Based on the analysis of vitals, the patient is classified as {triage_level}. Extracted key symptoms: {', '.join(extracted_symptoms)}. Directed to {specialty} department."
    
    return TriageResponse(
        triage_level=triage_level,
        urgency_score=urgency_score,
        reasoning=reasoning,
        recommended_facilities=recommended_facilities
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
