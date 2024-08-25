import streamlit as st
import pickle
import spacy

# Load the model
with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the vectorizer
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Preprocessing function
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

# Streamlit UI
st.title('Product Review Sentiment Analysis')

st.write("Enter a product review below, and click 'Predict Sentiment' to get the sentiment score.")

# Input text area for product review
review = st.text_area('Enter Product Review:', '')

if st.button('Predict Sentiment'):
    if review:
        # Preprocess input review
        processed_review = preprocess_text(review)
        
        # Vectorize input review
        review_vector = vectorizer.transform([processed_review])
        
        # Predict sentiment
        prediction = model.predict(review_vector)
        
        # Display prediction
        st.write(f'Sentiment Score: {prediction[0]}')
    else:
        st.write("Please enter a product review to get the sentiment score.")
