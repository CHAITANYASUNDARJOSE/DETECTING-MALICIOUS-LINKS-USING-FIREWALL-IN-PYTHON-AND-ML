import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from bs4 import BeautifulSoup
import requests

# Load your dataset (replace 'dataset.csv' with your dataset file)
data = pd.read_csv(r"C:\Users\\13thc\Downloads\jose\jose\urldata.csv")

# Preprocess the URLs
def preprocess_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract text from the HTML page (you can customize this part)
        text = soup.get_text()
        return text
    except:
        return ''

data['url'] = data['url'].apply(preprocess_url)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['url'], data['label'], test_size=0.2, random_state=42)

# Create a pipeline with TF-IDF and CountVectorizer
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),  
    ('tfidf', TfidfTransformer()), 
    ('classifier', LogisticRegression())
])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions on the test data
y_pred = pipeline.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Classify a new URL
def classify_url(url):
    text = preprocess_url(url)
    label = pipeline.predict([text])[0]
    return "Safe" if label == 0 else "Malicious"

# Example usage:
url_to_check = "https://example.com"
result = classify_url(url_to_check)
print(f'The URL "{url_to_check}" is {result}.')
