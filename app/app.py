import streamlit as st
import pickle
import re

# Load model & vectorizer
model = pickle.load(open("model/fake_news_model.pkl", "rb"))
tfidf = pickle.load(open("model/tfidf.pkl", "rb"))

# Matching the preprocessing used during training
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)     # remove URLs
    text = re.sub(r"[^a-zA-Z ]", "", text)  # remove symbols
    return text

st.set_page_config(page_title="Fake News Detection", layout="centered")

st.title("📰 Fake News Detection System")
st.write("Enter a news article or social media post below:")

user_input = st.text_area("News Text", height=200)

if st.button("Check News"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Clean the text exactly as we did in the notebook
        cleaned_input = clean_text(user_input)
        
        text_vec = tfidf.transform([cleaned_input])
        prob = model.predict_proba(text_vec)[0]
        real_probability = prob[1]

        import requests
        import urllib.parse

        # General AI Internet Check with LIVE WEB SCRAPING
        with st.spinner("🤖 AI is actively scraping the live internet (News & Wikipedia) for facts..."):
            context = ""
            
            # 1. Try fetching live news via DuckDuckGo
            try:
                from duckduckgo_search import DDGS
                news_results = DDGS().news(user_input, max_results=3)
                if news_results:
                    context += "Live News Articles:\n"
                    for res in news_results:
                        context += f"- {res['title']}: {res['body']}\n"
            except Exception:
                pass
                
            # 2. Try fetching Wikipedia for general knowledge
            try:
                import wikipedia
                import warnings
                warnings.filterwarnings("ignore", category=UserWarning, module='wikipedia')
                search_results = wikipedia.search(user_input, results=2)
                if search_results:
                    for page_title in search_results:
                        try:
                            summary = wikipedia.summary(page_title, sentences=2, auto_suggest=False)
                            context += f"\nWikipedia ({page_title}): {summary}\n"
                        except Exception:
                            pass
            except Exception:
                pass

            prompt = f"You are a strict fact-checking AI. Fact-check the user's statement. Here is the latest up-to-date internet context we scraped for you:\n{context}\n\nUser statement: '{user_input}'. Is it true or false? Start your response with exactly the word 'TRUE.', 'FALSE.', or 'UNVERIFIABLE.', followed by a clear explanation based ONLY on the provided internet context or your own general knowledge if the context is empty."
            
            try:
                url = "https://text.pollinations.ai/" + urllib.parse.quote(prompt)
                ai_response = requests.get(url, timeout=20).text.strip()
            except Exception:
                ai_response = "ERROR"

        st.markdown("---")
        st.subheader("🌐 General AI Fact-Check Result")
        
        if ai_response == "ERROR":
            st.error("Failed to reach the AI fact-checking server. Falling back to old model.")
        else:
            if ai_response.upper().startswith("TRUE"):
                st.success("✅ **REAL NEWS / TRUE FACT** (Verified by AI Agent)")
            elif ai_response.upper().startswith("FALSE"):
                st.error("🚨 **FAKE NEWS / FALSE FACT** (Verified by AI Agent)")
            else:
                st.warning("⚠️ **UNVERIFIABLE** (AI Agent couldn't find a definitive answer)")
                
            st.info(f"**AI Explanation:**\n\n{ai_response}")

        st.markdown("---")
        with st.expander("Legacy Political ML Model Score (For Comparison)", expanded=False):
            st.write(f"The original ISOT-based political classifier scored this as: **{real_probability*100:.2f}% Real**")
            st.caption("Note: The legacy ML model is not connected to the internet and only analyzes keywords, causing it to fail on general facts.")
