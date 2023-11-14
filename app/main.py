# from fastapi import FastAPI, File, UploadFile
# from pydantic import BaseModel
# from app.model.model import predict_pipeline
# from app.model.model import __version__ as model_version


# app = FastAPI()


# class AudioIn(BaseModel):
#     wav: UploadFile = File(...)

    
# class PredictionOut(BaseModel):
#     emotion: str

    
# @app.get("/")
# def home():
#     return {"health_check": "OK", "model_version": model_version}



# @app.post("/predict", response_model=PredictionOut)
# def predict(payload: UploadFile = File(...)):
#     emotions = predict_pipeline(payload)
#     return PredictionOut(emotion=emotions[0])
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
import requests

app = FastAPI()

# Replace with your Deepgram API Key
api_key = "313178ad7ceb92d4d389947590d93ae083bc743b"

# Specify the Deepgram API endpoint
api_url = "https://api.deepgram.com/v1/listen"


def get_deepgram_headers():
    return {"Authorization": f"Bearer {api_key}"}


@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...), headers: dict = Depends(get_deepgram_headers)):
    try:
        # Prepare the payload with the audio file
        files = {"content": (file.filename, file.file, file.content_type)}

        # Send the POST request to Deepgram API for transcription
        response = requests.post(api_url, headers=headers, files=files)

        # Check if the request was successful (HTTP status code 200)
        if response.status_code == 200:
            result = response.json()
            # Extract the transcript
            transcript = result["text"]
            return {"transcript": transcript}
        else:
            raise HTTPException(status_code=response.status_code, detail=f"Deepgram API Error: {response.text}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

