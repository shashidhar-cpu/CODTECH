import nltk
from nltk.corpus import movie_reviews
import random
import pandas as pd

nltk.download('movie_reviews')

# Load and shuffle data
docs = [(list(movie_reviews.words(fileid)), category)
        for category in movie_reviews.categories()
        for fileid in movie_reviews.fileids(category)]
random.shuffle(docs)

# Convert to DataFrame
texts = [' '.join(words) for words, _ in docs]
labels = [label for _, label in docs]

df = pd.DataFrame({'text': texts, 'label': labels})


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Encode labels: pos = 1, neg = 0
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Train model
model = LogisticRegression()
model.fit(X_train_vect, y_train)

# Predict
y_pred = model.predict(X_test_vect)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

