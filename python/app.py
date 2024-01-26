import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower();
    text = nltk.word_tokenize(text)
    result = []
    for i in text:
        if i.isalnum():
            result.append(i)

    text = result[:]
    result.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            result.append(i)
    text = result[:]
    result.clear()

    for i in text:
        result.append(ps.stem(i))

    return " ".join(result)
tfidf = pickle.load(open('vectorizer.pk1', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message: ")

if st.button('Predict'):

    transformed_sms = transform_text(input_sms)

    vector_input = tfidf.transform([transformed_sms])

    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("Spam Message!")
    else:
        st.header("Is not a Spam Message!")