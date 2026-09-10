import joblib
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title="Iris ML API")

model = joblib.load("models/model.pkl")


class PredictionRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest):
    prediction = model.predict([request.features])

    return {
        "prediction": int(prediction[0])
    }