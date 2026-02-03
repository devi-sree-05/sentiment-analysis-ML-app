from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from app.model import predict_sentiment

app = FastAPI(title="Sentiment Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request schema
class TextInput(BaseModel):
    text: str

# Health check
@app.get("/")
def read_root():
    return {"message": "Sentiment API is running"}

# Prediction endpoint
@app.post("/predict")
def predict(input: TextInput):
    result = predict_sentiment(input.text)
    return {
        "input_text": input.text,
        "sentiment": result["label"],
        "confidence": result["confidence"]
    }
