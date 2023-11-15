from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
import json
import requests

app = FastAPI()

# Replace with the new Deepgram API Key
api_key = "9458e94f9a1a422f6e3d87e575ccbd80f59fdb46"

# Specifying the Deepgram API endpoint
api_url = "https://api.deepgram.com/v1/listen"


@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(..., upload_max_size=50_000_000)):
    try:
        # Prepare the payload with the audio file
        files = {"content": (file.filename, file.file, file.content_type)}

        # # Specify Deepgram parameters
        # deepgram_params = {
        #     "model": "nova-2",
        #     "language": "en",
        #     "smart_format": True,
        # }

        # Specify Deepgram headers
        headers = {
             "accept": "application/json",
             "content-type": "application/json",
             "Authorization": f"Token {api_key}"
            }
        # Prepare the payload for Deepgram API
        payload = { "url": f"https://static.deepgram.com/examples/{file.filename}" }

        # Send the POST request to Deepgram API for transcription
        response = requests.post(api_url, headers=headers, json=payload)

        # Check if the request was successful (HTTP status code 200)
        response.raise_for_status()

        # Extract the transcript
        result = response.json()
        transcript = result.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("transcript", "")

        return {"transcript": transcript}

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Request to Deepgram API failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
