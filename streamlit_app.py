import streamlit as st

st.set_page_config(page_title="AI Sentiment Analysis", layout="centered")

st.title("AI Sentiment Analysis App")
st.write("Enter text below and click Analyze. Uses a Hugging Face sentiment model if available, otherwise falls back to TextBlob.")


# Try to use Hugging Face transformers; if not installed, fallback to TextBlob
USE_TRANSFORMERS = False
try:
    from transformers import pipeline
    USE_TRANSFORMERS = True
except Exception:
    USE_TRANSFORMERS = False

@st.cache_resource
def load_transformer_pipeline():
    try:
        return pipeline("sentiment-analysis")
    except Exception:
        return None


transformer_pipeline = None
if USE_TRANSFORMERS:
    transformer_pipeline = load_transformer_pipeline()
# Always try to import TextBlob as a local lightweight fallback
try:
    from textblob import TextBlob
except Exception:
    TextBlob = None

# If transformers was available but pipeline failed to load, we'll fall back to TextBlob below


def analyze_with_transformers(text: str):
    if transformer_pipeline is None:
        return {"label": "ERROR", "score": 0.0}
    out = transformer_pipeline(text)[0]
    return out


def analyze_with_textblob(text: str):
    if TextBlob is None:
        return {"label": "ERROR", "score": 0.0}
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.05:
        label = "POSITIVE"
    elif polarity < -0.05:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"
    # map polarity (-1..1) to 0..1 roughly
    score = abs(polarity)
    return {"label": label, "score": float(score)}


# Lightweight lexicon-based fallback (no external deps)
LEXICON = {
    "good": 1.0, "great": 1.5, "excellent": 2.0, "happy": 1.0, "love": 1.5,
    "bad": -1.0, "terrible": -2.0, "sad": -1.0, "hate": -1.5, "awful": -2.0
}


def analyze_with_lexicon(text: str):
    words = [w.strip(".,!?:;\"'()[]") .lower() for w in text.split()]
    score = 0.0
    count = 0
    for w in words:
        if w in LEXICON:
            score += LEXICON[w]
            count += 1
    if count == 0:
        return {"label": "NEUTRAL", "score": 0.0}
    avg = score / count
    if avg > 0.1:
        label = "POSITIVE"
    elif avg < -0.1:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"
    # map avg magnitude to 0..1 roughly
    conf = min(1.0, abs(avg) / 2.0)
    return {"label": label, "score": float(conf)}


user_input = st.text_area("Enter text to analyze sentiment:", height=160)

if st.button("Analyze"):
    if not user_input:
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            # Prefer transformer pipeline if available and loaded; otherwise use TextBlob then lexicon fallback
            if USE_TRANSFORMERS and transformer_pipeline is not None:
                result = analyze_with_transformers(user_input)
            elif TextBlob is not None:
                result = analyze_with_textblob(user_input)
            else:
                # Use lexicon-based fallback (no external deps)
                result = analyze_with_lexicon(user_input)

        label = result.get("label")
        score = result.get("score", 0.0)
        if label == "POSITIVE":
            st.success(f"Positive — Confidence: {score:.2%}")
        elif label == "NEGATIVE":
            st.error(f"Negative — Confidence: {score:.2%}")
        elif label == "NEUTRAL":
            st.info(f"Neutral — Confidence: {score:.2%}")
        else:
            st.warning("Model error or missing dependency: see console for details.")
        st.write("Model output:", result)
