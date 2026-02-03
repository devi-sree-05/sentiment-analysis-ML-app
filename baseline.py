import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. Load data
train_df = pd.read_csv(
    "dataset/archive/train.csv",
    encoding="latin-1"
)

# 2. Keep only required columns
train_df = train_df[["text", "sentiment"]]

# Drop missing values (simple & safe)
train_df = train_df.dropna()

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    train_df["text"],
    train_df["sentiment"],
    test_size=0.2,
    random_state=42,
    stratify=train_df["sentiment"]
)

# 4. TF-IDF vectorization
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 5. Train baseline model
model = LogisticRegression(
    max_iter=1000,
    n_jobs=-1
)

model.fit(X_train_tfidf, y_train)

# 6. Evaluate
y_pred = model.predict(X_test_tfidf)

print("\n=== BASELINE MODEL PERFORMANCE ===\n")
print(classification_report(y_test, y_pred))
