from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Cargar el modelo desde el archivo .pkl
model = joblib.load("diabetes_model.pkl")

# Definición de la clase InputData
class InputData(BaseModel):
    pregnancies: int
    glucose: int
    blood_pressure: int
    skin_thickness: int
    insulin: int
    bmi: float
    diabetes_pedigree_function: float
    age: int
# Configuración de CORS
origins = ["*"]  # Puedes ajustar esto según tus necesidades
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Agrega OPTIONS aquí
    allow_headers=["*"],
)
# Endpoint para realizar la predicción
@app.post("/predict_diabetes")
def predict_diabetes(data: InputData):
    try:
        # Convertir datos de entrada a un array NumPy
        input_features = np.array([[
            data.pregnancies, data.glucose, data.blood_pressure,
            data.skin_thickness, data.insulin, data.bmi,
            data.diabetes_pedigree_function, data.age
        ]])

        # Realizar la predicción
        prediction = model.predict(input_features)

        # Convertir la predicción a formato booleano (tiene diabetes o no)
        has_diabetes = bool(prediction[0])

        return {"has_diabetes": has_diabetes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
