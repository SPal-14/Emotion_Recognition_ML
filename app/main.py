from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from app.model.model import predict_pipeline
from app.model.model import __version__ as model_version


app = FastAPI()


class AudioIn(BaseModel):
    wav: UploadFile = File(...)

    
class PredictionOut(BaseModel):
    emotion: str

    
@app.get("/")
def home():
    return {"health_check": "OK", "model_version": model_version}



@app.post("/predict", response_model=PredictionOut)
def predict(payload: UploadFile = File(...)):
    emotions = predict_pipeline(payload)
    return PredictionOut(emotion=emotions[0])


