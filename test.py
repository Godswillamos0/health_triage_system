import requests
res = requests.post("http://127.0.0.1:8000/api/triage", json={"age": 35, "heart_rate": 85, "systolic_blood_pressure": 120, "oxygen_saturation": 98.5, "body_temperature": 37.0, "pain_level": 3, "chronic_disease_count": 0, "previous_er_visits": 0, "symptoms": "headache", "latitude": 7.2507, "longitude": 5.2069})
print(res.status_code)
print(res.text)
