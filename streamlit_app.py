import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime

st.set_page_config(page_title="AI Sentiment Analysis", layout="centered")

st.title("AI Sentiment Analysis App")
st.write("Enter text below and submit for sentiment analysis. Choose a model in the sidebar.")

# Sidebar and model selector are initialized after determining available backends


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

# Model selector in the sidebar (after availability detection)
model_options = ["Auto (best available)", "Transformers", "TextBlob", "Lexicon"]
model_choice = st.sidebar.selectbox("Model", model_options, index=0)
st.sidebar.write("Available: ", "Transformers" if USE_TRANSFORMERS else "Transformers (not installed)")

# Help and export in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("Use the form below to type text and submit. History is stored in this session.")
if "history" in st.session_state and st.session_state.history:
    df_hist = pd.DataFrame(st.session_state.history)
    csv = df_hist.to_csv(index=False)
    st.sidebar.download_button("Download history CSV", csv, file_name="sentiment_history.csv")


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


with st.form("input_form"):
    user_input = st.text_area("Enter text to analyze sentiment:", height=160)
    submitted = st.form_submit_button("Analyze")


@st.cache_data
def cached_analyze(text: str, mode: str):
    # mode: one of model_options
    mode = mode or "Auto (best available)"
    # Resolve actual mode
    if mode.startswith("Auto"):
        if USE_TRANSFORMERS and transformer_pipeline is not None:
            actual = "Transformers"
        elif TextBlob is not None:
            actual = "TextBlob"
        else:
            actual = "Lexicon"
    else:
        actual = mode

    if actual == "Transformers":
        return analyze_with_transformers(text)
    elif actual == "TextBlob":
        return analyze_with_textblob(text)
    else:
        return analyze_with_lexicon(text)

if submitted:
    if not user_input:
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            result = cached_analyze(user_input, model_choice)

        label = result.get("label")
        score = float(result.get("score", 0.0))
        # Which actual model was used
        used_model = model_choice
        if model_choice.startswith("Auto"):
            if USE_TRANSFORMERS and transformer_pipeline is not None:
                used_model = "Transformers"
            elif TextBlob is not None:
                used_model = "TextBlob"
            else:
                used_model = "Lexicon"
        if label == "POSITIVE":
            st.success(f"Positive — Confidence: {score:.2%}")
        elif label == "NEGATIVE":
            st.error(f"Negative — Confidence: {score:.2%}")
        elif label == "NEUTRAL":
            st.info(f"Neutral — Confidence: {score:.2%}")
        else:
            st.warning("Model error or missing dependency: see console for details.")
        st.write("Model output:", result)

        # Record history in session state
        if "history" not in st.session_state:
            st.session_state.history = []
        st.session_state.history.append({
            "time": datetime.now(),
            "text": user_input,
            "label": label,
            "score": score,
            "model": used_model,
        })

        # Visualizations
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Result")
            st.markdown(f"**Sentiment:** {label}")
            st.markdown(f"**Confidence:** {score:.2%}")
            # confidence bar using Altair
            conf_df = pd.DataFrame({"score": [score]})
            bar = alt.Chart(conf_df).mark_bar(size=40).encode(
                x=alt.X("score:Q", scale=alt.Scale(domain=(0, 1)), title=None),
                color=alt.condition(alt.datum.score > 0.5, alt.value("#16a34a"), alt.value("#ef4444"))
            )
            st.altair_chart(bar, use_container_width=True)
            st.caption(f"Model used: {used_model}")
        with col2:
            st.subheader("Quick metrics")
            st.metric("Sentiment", label)
            st.metric("Confidence", f"{score:.2%}")

        # Prepare history dataframe
        if st.session_state.history:
            df = pd.DataFrame(st.session_state.history)
            # Label distribution
            st.subheader("Label distribution")
            dist = df["label"].value_counts().reset_index()
            dist.columns = ["label", "count"]
            chart = alt.Chart(dist).mark_bar().encode(
                x=alt.X("label:N", sort=["POSITIVE", "NEUTRAL", "NEGATIVE"]),
                y="count:Q",
                color="label:N",
            )
            st.altair_chart(chart, use_container_width=True)

            # Confidence over time
            st.subheader("Confidence over time")
            df_time = df.copy()
            df_time["time"] = pd.to_datetime(df_time["time"]) 
            line = alt.Chart(df_time).mark_line(point=True).encode(
                x=alt.X("time:T", title="Time"),
                y=alt.Y("score:Q", title="Confidence"),
                color=alt.Color("label:N")
            )
            st.altair_chart(line, use_container_width=True)

            st.subheader("History")
            st.dataframe(df[["time", "model", "label", "score", "text"]].sort_values(by="time", ascending=False))
