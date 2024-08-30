import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Load model and vectorizer
with open('model1.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Initialize Streamlit app
st.set_page_config(page_title="Sentiment Analysis", page_icon=":bar_chart:", layout="wide")

# Title and Description
st.title('📊 Sentiment Analysis of Product Reviews')
st.markdown("""
Enter a product review below to analyze its sentiment. The sentiment can be **Positive**, **Negative**, or **Neutral**.
""")

# Create a container for the review input
with st.container():
    st.subheader('📝 Review Input')
    review = st.text_area("Paste your review here:", height=150)

    # Create a button to analyze the review
    if st.button('🔍 Analyze'):
        if review:
            # Preprocess review
            def preprocess_text(text):
                # Convert text to lowercase
                text = text.lower()
            
                # Remove URLs
                text = re.sub(r'http\S+|www\S+|https\S+', '', text)
            
                # Remove user mentions (e.g., @username)
                text = re.sub(r'@\w+', '', text)
            
                # Remove special characters and punctuation (keeping only alphanumeric characters and spaces)
                text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
            
                # Remove extra spaces
                text = re.sub(r'\s+', ' ', text).strip()
            
                return text

            cleaned_review = preprocess_text(review)
            review_vector = vectorizer.transform([cleaned_review])
            sentiment = model.predict(review_vector)[0]
           
            sentiment_label = (
                'Positive' if sentiment == 2 
                else 'Negative' if sentiment == 0
                else 'Neutral'
            )

            # Display result
            st.subheader('🔍 Analysis Result:')
            st.write(f'**The sentiment of the review is: {sentiment_label}**')
        else:
            st.warning("Please enter a review before analyzing.")
