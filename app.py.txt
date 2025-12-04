from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Cargar modelo
model = joblib.load("sleep_model.pkl")

# Definir datos de entrada
class InputData(BaseModel):
    Anxiety: float
    Depression: float
    Stress: float
    SleepHours: float
    PhysicalActivity: float
    SelfEsteem: float
    SocialSupport: float
    AlcoholUse: float
    SuicidalIdeation: float
    AcademicPerformance: float
    Age: float

@app.post("/predict")
def predict(data: InputData):
    values = np.array([[data.Anxiety, data.Depression, data.Stress,
                        data.SleepHours, data.PhysicalActivity,
                        data.SelfEsteem, data.SocialSupport,
                        data.AlcoholUse, data.SuicidalIdeation,
                        data.AcademicPerformance, data.Age]])
    
    prediction = int(model.predict(values)[0])
    
    return {"prediction": prediction}
