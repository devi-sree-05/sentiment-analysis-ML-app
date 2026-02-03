from transformers import pipeline

# Load pretrained sentiment analysis pipeline
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Test sentences
sentences = [
    "I love this movie so much!",
    "This is the worst experience ever",
    "It was okay, nothing special"
]

# Run inference
results = sentiment_pipeline(sentences)

for text, result in zip(sentences, results):
    print(f"Text: {text}")
    print(f"Prediction: {result['label']}, Confidence: {result['score']:.2f}")
    print("-" * 40)
