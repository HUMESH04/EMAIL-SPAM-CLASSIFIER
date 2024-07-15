import streamlit as st
import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load the saved vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Initialize the PorterStemmer
ps = PorterStemmer()

# Define the text preprocessing function
def transform_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', ' ', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize text
    words = text.split()
    # Remove stop words
    words = [word for word in words if word not in stopwords.words('english')]
    # Stem words
    words = [ps.stem(word) for word in words]
    return ' '.join(words)

# Streamlit app
st.title("Email Spam Classifier")
input_sms = st.text_area("Enter message")

if st.button('Predict'):
    # Preprocess the input text
    transformed_sms = transform_text(input_sms)
    # Vectorize the preprocessed text
    vector_input = tfidf.transform([transformed_sms])
    # Predict using the model
    result = model.predict(vector_input)[0]
    # Display the result
    if result == 0:
        st.header("not spam")
    else:
        st.header("spam")