import webbrowser
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib
import firewall

def redirect_to_url():
    url = input("Enter the URL: ")
    if firewall.is_malicious_url(url):
        print("Access to the URL is restricted due to security concerns.")
    else:
        webbrowser.open(url)

redirect_to_url()
