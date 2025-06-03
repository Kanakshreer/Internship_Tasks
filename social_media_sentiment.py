import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Download NLTK data (only once)
nltk.download('punkt')

# Preprocess function to clean and tokenize text
def preprocess(text):
    if not isinstance(text, str):
        return ""  # Return empty string for NaN or non-string values
    text = text.lower()
    tokens = word_tokenize(text)
    # You can add more cleaning steps here if you want (remove stopwords, punctuation, etc.)
    return " ".join(tokens)

# Load your dataset
df = pd.read_csv(r'C:\Users\Kanak Shree\OneDrive\Desktop\FutureIntern\task2\tweets.csv')
  

# Drop rows where 'text' is missing
df = df.dropna(subset=['text'])

# Apply preprocessing on text column
df['clean_text'] = df['text'].apply(preprocess)

# Features and labels
X = df['clean_text']
y = df['sentiment']  # Assuming sentiment column is named 'sentiment'

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text data
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train classifier
#model = LogisticRegression()
model = LogisticRegression(max_iter=1000)

model.fit(X_train_vec, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_vec)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the model and vectorizer
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("Model and vectorizer saved successfully.")
