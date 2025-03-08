import streamlit as st
import pickle
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download required datasets (remove incorrect one)
nltk.download('punkt')
nltk.download('stopwords')

# Initialize stemmer and stopwords set
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))  # Store stopwords in a set for faster lookup

# Function to preprocess text
def transform_text(text):
    text = text.lower()
    text = word_tokenize(text)  # Use the correct tokenizer

    # Remove non-alphanumeric characters
    text = [i for i in text if i.isalnum()]

    # Remove stopwords and punctuation
    text = [i for i in text if i not in stop_words and i not in string.punctuation]

    # Apply stemming
    text = [ps.stem(i) for i in text]

    return " ".join(text)

# Load the vectorizer and model with error handling
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Ensure 'vectorizer.pkl' and 'model.pkl' exist.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Streamlit UI
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if input_sms.strip():  # Ensure input is not empty
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])  # Vectorize input
        result = model.predict(vector_input)[0]  # Predict

        # Display result
        st.header("Spam" if result == 1 else "Not Spam")
    else:
        st.warning("Please enter a message before predicting.")
