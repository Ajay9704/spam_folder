import streamlit as st
import pickle
import string
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure required NLTK datasets are downloaded
nltk.data.path.append(os.path.join(os.getcwd(), "nltk_data"))
nltk.download('punkt', download_dir=os.path.join(os.getcwd(), "nltk_data"))
nltk.download('stopwords', download_dir=os.path.join(os.getcwd(), "nltk_data"))
nltk.download('wordnet', download_dir=os.path.join(os.getcwd(), "nltk_data"))
nltk.download('omw-1.4', download_dir=os.path.join(os.getcwd(), "nltk_data"))
nltk.download('averaged_perceptron_tagger', download_dir=os.path.join(os.getcwd(), "nltk_data"))
nltk.download('maxent_ne_chunker', download_dir=os.path.join(os.getcwd(), "nltk_data"))
nltk.download('words', download_dir=os.path.join(os.getcwd(), "nltk_data"))

ps = PorterStemmer()

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric tokens
    text = [i for i in text if i.isalnum()]

    # Remove stopwords and punctuation
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]

    # Apply stemming
    text = [ps.stem(i) for i in text]

    return " ".join(text)

# Load vectorizer and model with error handling
try:
    with open('vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Error: Model or vectorizer file not found. Please check your deployment.")
    st.stop()

# Verify that tfidf is fitted
from sklearn.utils.validation import check_is_fitted
try:
    check_is_fitted(tfidf)
except:
    st.error("Error: The TF-IDF vectorizer is not fitted. Re-train and save it before using the app.")
    st.stop()

# Streamlit UI
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter a message before predicting.")
    else:
        # 1. Preprocess the input text
        transformed_sms = transform_text(input_sms)

        # 2. Convert to TF-IDF vector
        vector_input = tfidf.transform([transformed_sms])

        # 3. Predict spam or not
        result = model.predict(vector_input)[0]

        # 4. Display result
        st.header("Spam" if result == 1 else "Not Spam")
