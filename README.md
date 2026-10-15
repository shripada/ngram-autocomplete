# N-gram Autocomplete

A Streamlit app that suggests the next word as you type, using a word-level
n-gram model (up to 4-grams, stupid backoff) trained on 8 public-domain
Project Gutenberg books (~1M words).

```bash
uv sync
uv run streamlit run app.py
```

The first run downloads the books into `data/` (about 6 MB) and takes ~20 s.
Later runs start in a few seconds.

- `corpus.py` – downloads and cleans the books
- `ngram.py` – counts n-grams and predicts the next word
- `app.py` – the Streamlit UI (dropdown via `streamlit-searchbox`)
