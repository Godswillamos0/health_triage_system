# Decision Support System (IDSS) for Symptom Triage & Healthcare Facility Recommendation

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/FastAPI-0.100+-00a393.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/XGBoost-Machine_Learning-orange.svg" alt="XGBoost">
</div>

---

## 📖 Project Overview

This project was proposed by **Adedeji Adedunmola ‘Deniolabomi** from the Department of Information Systems, Federal University of Technology, Akure (FUTA), and supervised by **Dr. A.I Makinde**.

The system is an end-to-end **Decision Support System (IDSS)** designed to bridge the gap in emergency room operations and healthcare facility allocation. Traditional triage methods rely heavily on human judgment, achieving roughly 60% accuracy on average, which leads to prolonged wait times and hospital overcrowding. Furthermore, in developing regions like Nigeria, the uneven distribution of healthcare facilities makes it difficult for patients to find the right care quickly.

This solution integrates Natural Language Processing (NLP), Machine Learning (XGBoost), and Multi-Criteria Decision Making (MCDM) to provide accurate triage predictions and instantly route patients to the most appropriate, nearest hospital with available bed capacity.

---

## ✨ Key Features

1. **Symptom Checker / NLP Engine**  
   Extracts, tokenizes, and normalizes user-provided symptom data to understand the chief complaint effectively.

2. **XGBoost Machine Learning Predictor**  
   Analyzes structured patient data (age, vital signs, pain level, medical history) to instantly predict the appropriate triage level (`Routine`, `Urgent`, `Emergency`, `Resuscitation`). XGBoost was specifically chosen for its speed, handling of tabular data, and high degree of explainability compared to deep learning "black box" models.

3. **Hospital Ranker & Facility Recommender**  
   Uses the **Haversine formula** to calculate the geographic distance between the patient's coordinates and nearby hospitals. An **MCDM (Multi-Criteria Decision Making)** scoring function then ranks facilities based on proximity, bed availability, and matching medical specialties.

4. **Dynamic User Interface**  
   A beautiful, responsive web interface built with modern design principles (Glassmorphism, glowing micro-animations, dark mode aesthetics) allowing patients or medical staff to easily input data and view actionable recommendations.

5. **Knowledge Base Layer**  
   Backed by synthetic tabular datasets (`synthetic_medical_triage.csv`) and localized hospital coordinates (`hospitals.csv`), acting as the primary knowledge store for the system.

---

## 🛠️ Technology Stack

- **Backend Framework:** FastAPI, Uvicorn, Python
- **Machine Learning:** XGBoost, Scikit-learn, Pandas, Joblib
- **Frontend / Templating:** HTML5, Vanilla CSS3, JavaScript, Jinja2
- **Package Manager:** `uv` (Fast Python package manager)

---

## 🚀 Getting Started

### Prerequisites

You need [Python](https://www.python.org/downloads/) (3.11 or higher) and [uv](https://github.com/astral-sh/uv) installed on your machine.

### Installation & Setup

1. **Clone the repository** (or navigate to the project directory):
   ```bash
   cd triage_system
   ```

2. **Install dependencies** (handled automatically by `uv`):
   ```bash
   uv sync
   ```
   *(Note: The project uses `pyproject.toml` and `uv.lock` for strict dependency management).*

3. **Pre-train the Model (Optional but Recommended)**  
   The system will automatically train the XGBoost model on the first run if it doesn't exist, but you can explicitly build it via:
   ```bash
   uv run python -c "from app.services.predictor import predictor_service; predictor_service.initialize_model()"
   ```

4. **Run the Server**:
   ```bash
   uv run python main.py
   ```
   Alternatively, you can run the ASGI server directly:
   ```bash
   uv run uvicorn main:app --host 127.0.0.1 --port 8000 --reload
   ```

5. **Access the Application**:  
   Open your browser and navigate to:  
   👉 **http://127.0.0.1:8000**

---

## 📁 Project Structure

```text
triage_system/
│
├── app/
│   ├── api/            # API routers (if expanded later)
│   ├── core/           # Core configuration and settings
│   ├── data/           # Knowledge base: CSV datasets and trained model files
│   ├── schemas/        # Pydantic models for data validation (triage.py)
│   ├── services/       # Core business logic (NLP, Predictor, Recommender)
│   ├── templates/      # Jinja2 HTML templates (index.html)
│   └── utils/          # Helper functions
│
├── main.py             # FastAPI entry point & API routing
├── pyproject.toml      # Project metadata and dependencies
├── uv.lock             # Exact dependency locking
└── README.md           # Project documentation
```

---

## 🎯 Expected Contributions & Impact

- **Healthcare Democratization:** Adapts complex ML models into resource-limited environments using strictly free and open-source tools.
- **Model Transparency:** Replaces opaque algorithms with explainable, feature-driven predictions.
- **Prototype Innovation:** Serves as a foundational, open-source springboard for future Nigerian Health-Tech developers to expand upon.

---

<p align="center">
  <i>Developed with ❤️ for the advancement of accessible healthcare technology.</i>
</p>
