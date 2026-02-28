
# AI Sentiment Analysis (Streamlit)

- **Purpose:** small Streamlit app that accepts user text and shows sentiment.
- **How it works:** prefers a Hugging Face `sentiment-analysis` pipeline; if `transformers` isn't available it falls back to `TextBlob`; if that isn't installed it uses a small built-in lexicon.
- **Model notes:** the agent that built this repo used `GPT-5 mini` to help author code and commits. The app itself uses the Hugging Face default sentiment model (e.g. `distilbert-base-uncased-finetuned-sst-2-english`) when `transformers` is available.
- **Run locally:**

  ```bash
  pip install -r requirements.txt
  python -m streamlit run streamlit_app.py
  ```

- **Deployment:** push to GitHub (done to `main`) and deploy on Streamlit Community Cloud by connecting this repository. The app is ready for deployment; Streamlit will install `requirements.txt` and run `streamlit_app.py`.



