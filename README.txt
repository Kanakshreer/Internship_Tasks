# Social Media Sentiment Analysis

This project performs sentiment analysis on social media posts (e.g., tweets) using machine learning techniques. It classifies text into categories like **Positive**, **Negative**, **Neutral**, and **Irrelevant** based on the sentiment expressed.

## 🧠 Project Description

The goal of this project is to gain insights from social media text by classifying sentiments. It involves:
- Cleaning and preprocessing raw text data
- Converting text into numerical features using vectorization
- Training a sentiment classifier using machine learning
- Evaluating model performance with standard metrics

## ⚙️ Techniques Used

- **Text Preprocessing:** Lowercasing, punctuation removal, tokenization
- **Vectorization:** CountVectorizer or TF-IDF
- **Modeling:** Logistic Regression (can be replaced with Naive Bayes, SVM, RNN, or Transformers)
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score

## 🗂️ Project Structure

```
├── social_media_sentiment.py   # Main script for training and evaluation
├── tweets.csv                  # Dataset with text and sentiment labels
├── model.pkl                   # Saved trained sentiment classifier
├── vectorizer.pkl              # Saved text vectorizer
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── .gitignore                  # Git ignored files
```

## 📊 Results

The model achieved an **accuracy of ~78.4%** on the test set. Below is the performance summary:

| Sentiment   | Precision | Recall | F1-Score |
|-------------|-----------|--------|----------|
| Irrelevant  | 0.81      | 0.68   | 0.74     |
| Negative    | 0.79      | 0.84   | 0.81     |
| Neutral     | 0.78      | 0.75   | 0.77     |
| Positive    | 0.77      | 0.82   | 0.80     |

## 🛠️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/social-media-sentiment-analysis.git
cd social-media-sentiment-analysis
pip install -r requirements.txt
```

## 🚀 Usage

Run the main script:

```bash
python social_media_sentiment.py
```

Make sure `tweets.csv` is in the same directory.

## 📌 Dependencies

- pandas  
- numpy  
- scikit-learn  
- nltk  

## 📄 License

This project is for educational purposes and is not licensed for commercial use.

---

**Author:** [Kanakshree Rathore]  
