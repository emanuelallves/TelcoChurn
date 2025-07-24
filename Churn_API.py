from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import mlflow.sklearn
import mlflow
from mlflow.tracking import MlflowClient

app = FastAPI()
mlflow.set_tracking_uri('http://127.0.0.1:5000/')

client = MlflowClient()
model_versions = client.search_model_versions("name='Telco_Churn'")
version = max([int(v.version) for v in model_versions])
model = mlflow.sklearn.load_model(f'models:/Telco_Churn/{version}')

class PredictRequest(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str

@app.post('/predict')
def predict(requests: List[PredictRequest]):
    data = [r.dict() for r in requests]
    df = pd.DataFrame(data)
    predictions = model.predict(df)
    return {"predictions": predictions.tolist()}

@app.get('/')
def read_root():
    return {'message': 'Welcome to the Churn Classification API'}
