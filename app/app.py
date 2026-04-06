import streamlit as st
import pickle

# Load model & vectorizer
model = pickle.load(open("model/fake_news_model.pkl", "rb"))
tfidf = pickle.load(open("model/tfidf.pkl", "rb"))

st.set_page_config(page_title="Fake News Detection", layout="centered")

st.title("📰 Fake News Detection System")
st.write("Enter a news article or social media post below:")

user_input = st.text_area("News Text", height=200)

if st.button("Check News"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        text_vec = tfidf.transform([user_input])
        prediction = model.predict(text_vec)

        if prediction[0] == 0:
            st.error("🚨 This looks like FAKE news")
        else:
            st.success("✅ This looks like REAL news")
