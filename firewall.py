import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib

def train_model():
    # Load the dataset of malicious URLs
    df = pd.read_csv(r'C:\Users\13thc\Downloads\urldata.csv')
    urls = df['url']
    labels = df['label'] 
    
    # Split the dataset into training and testing sets
    train_urls, test_urls, train_labels, test_labels = train_test_split(urls, labels, test_size=0.2, random_state=42)
    
    # Create a TF-IDF vectorizer to convert URLs into numerical features
    vectorizer = TfidfVectorizer()
    train_features = vectorizer.fit_transform(train_urls)
    
    # Train a Support Vector Machine (SVM) classifier
    model = SVC()
    model.fit(train_features, train_labels)
    
    # Save the trained model and vectorizer for future use
    joblib.dump(model, 'firewall_model.joblib')
    joblib.dump(vectorizer, 'url_vectorizer.joblib')

def is_malicious_url(url):
    # Load the trained model and vectorizer
    model = joblib.load('firewall_model.joblib')
    vectorizer = joblib.load('url_vectorizer.joblib')
    
    # Transform the input URL into numerical features
    url_features = vectorizer.transform([url])
    
    # Use the trained model to predict if the URL is malicious or not
    prediction = model.predict(url_features)
    
    # Return True if the URL is predicted to be malicious, False otherwise
    return bool(prediction[0])