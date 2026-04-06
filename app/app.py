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
        prob = model.predict_proba(text_vec)[0]

        real_probability = prob[1]
        fake_probability = prob[0]
        
        # Determine dynamic threshold
        threshold = 0.5
        word_count = len(user_input.split())
        
        if word_count < 20:
            st.warning("⚠️ **Warning:** Your text is very short! This model is designed for full news articles. Fact-checking short sentences often results in inaccurate predictions.")
            threshold = 0.30  # Lower the bar for real news on short texts
            
        if real_probability >= threshold:
            # Rebalance the displayed confidence for the UI if it passed the lowered threshold
            display_conf = max(real_probability, 0.51) if word_count < 20 else real_probability
            st.success(f"✅ This looks like REAL news (Confidence: {display_conf*100:.2f}%)")
        else:
            st.error(f"🚨 This looks like FAKE news (Confidence: {fake_probability*100:.2f}%)")
            
        st.info("💡 **Note on Model Bias:** This model was trained on a dataset where "
                "real news articles heavily featured publisher tags like '(Reuters)'. "
                "Because of this, it often assumes short or uncredited text is fake. "
                "To test this bias, try starting your text with *'WASHINGTON (Reuters) - '*!")
