
# AI Sentiment Analysis (Streamlit)

- **Purpose:** small Streamlit app that accepts user text and shows sentiment.
- **How it works:** prefers a Hugging Face `sentiment-analysis` pipeline; if `transformers` isn't available it falls back to `TextBlob`; if that isn't installed it uses a small built-in lexicon.
- **Model notes:** the agent that built this repo used `GPT-5 mini` to help author code and commits. The app itself uses the Hugging Face default sentiment model (e.g. `distilbert-base-uncased-finetuned-sst-2-english`) when `transformers` is available.

  ```bash
  pip install -r requirements.txt
  python -m streamlit run streamlit_app.py
  ```

 
 - **Visualizations added:** current sentiment metrics, a confidence progress bar, label-distribution bar chart, confidence-over-time line chart, and a searchable history table.
 - **Interactive UI updates:** model selector (Auto / Transformers / TextBlob / Lexicon), `st.form` submission, cached predictions, and history CSV export.



