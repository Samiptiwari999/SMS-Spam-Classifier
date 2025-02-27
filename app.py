import pickle
import streamlit as st
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

vector = pickle.load(open('vector.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

st.title("Spam Email Classifier")
input_mail = st.text_area("Enter Message")

if st.button("Predict"):
    transformed = transform_text(input_mail)
    vectorized = vector.transform([transformed]).toarray()
    prediction = model.predict(vectorized)[0]
    if prediction==1:
        st.header("Spam")
    else:
        st.header("Not Spam")