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
        prob = model.predict_proba(text_vec)[0]

        if prediction[0] == 0:
            st.error(f"🚨 This looks like FAKE news (Confidence: {prob[0]*100:.2f}%)")
        else:
            st.success(f"✅ This looks like REAL news (Confidence: {prob[1]*100:.2f}%)")
            
        st.info("💡 **Note on Model Bias:** This model was trained on a dataset where "
                "real news articles heavily featured publisher tags like '(Reuters)'. "
                "Because of this, it often assumes short or uncredited text is fake. "
                "To test this bias, try starting your text with *'WASHINGTON (Reuters) - '*!")
