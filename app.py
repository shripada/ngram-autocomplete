"""Next-word autocomplete powered by a word n-gram model.

Run with:  uv run streamlit run app.py
"""

import streamlit as st
from streamlit_searchbox import st_searchbox

from corpus import BOOKS, load_corpus
from ngram import BOS, NGramModel

MAX_N = 4

st.set_page_config(page_title="N-gram Autocomplete", page_icon="⌨️", layout="centered")


@st.cache_resource(show_spinner="Downloading books and counting n-grams (first run only)…")
def get_model() -> NGramModel:
    return NGramModel(n=MAX_N).fit(load_corpus(BOOKS))


model = get_model()

with st.sidebar:
    st.header("Settings")
    order = st.slider(
        "n (n-gram size)",
        min_value=1,
        max_value=MAX_N,
        value=MAX_N,
        help="How many words the model looks at: n=1 ignores context, "
        "n=2 uses the last word, n=3 the last two, n=4 the last three.",
    )
    top_k = st.slider("Suggestions to show", 3, 10, 5)
    st.divider()
    st.caption(f"Corpus: {len(BOOKS)} Project Gutenberg books, {model.num_tokens:,} words")
    for title in BOOKS.values():
        st.caption(f"• {title}")


def styled(word: str, at_sentence_start: bool) -> str:
    if word == "i" or word.startswith("i'"):
        return "I" + word[1:]
    return word.capitalize() if at_sentence_start else word


def suggest(text: str, remember: bool = True) -> list[str]:
    suggestions, info = model.predict(text, k=top_k, order=order)
    if remember:
        st.session_state["last"] = (text, suggestions, info)

    prefix = info["prefix"]
    stem = text[: len(text) - len(prefix)] if prefix else text
    if stem and not stem[-1].isspace():
        stem += " "
    at_start = not stem.strip() or stem.rstrip()[-1] in ".!?"
    return [stem + styled(word, at_start) + " " for word, *_ in suggestions]


st.title("⌨️ N-gram Autocomplete")
st.write(
    "Start typing. The dropdown suggests the next word, or finishes the word "
    "you're typing. Pick one to add it, then keep typing."
)

st_searchbox(
    suggest,
    placeholder="Type here, e.g. 'It was the' or 'I do not'",
    key="ngram_box",
    edit_after_submit="option",
    default_options=suggest("", remember=False),
    debounce=100,
)

last = st.session_state.get("last")
if last and last[0]:
    text, suggestions, info = last
    looked_at = [w for w in info["context"] if w != BOS]

    st.subheader("Why these suggestions?")
    if order == 1:
        line = "With n = 1 the model ignores your words and suggests the most common words in the books"
    elif looked_at:
        line = f"The model looks only at your last words: **{' '.join(looked_at)}**"
    else:
        line = "You are at the start of a sentence, so the model looks for common first words"
    if info["prefix"]:
        line += f", and keeps only words starting with **{info['prefix']}**"
    st.markdown(line + ".")

    if suggestions:
        rows = []
        for word, _, ctx, count in suggestions:
            before = " ".join(w for w in ctx if w != BOS)
            if ctx and all(w == BOS for w in ctx):
                before = "(start of sentence)"
            rows.append({"suggestion": word, "found in the books as": f"{before} {word}".strip(), "times seen": count})
        st.dataframe(rows, hide_index=True)
        if any(len([w for w in ctx if w != BOS]) < len(looked_at) for _, _, ctx, _ in suggestions):
            st.caption(
                "When a phrase is shorter than your last words, the longer phrase "
                "never appeared in the books. The model dropped words from the front "
                "and searched again."
            )
    else:
        st.info("No word in the books starts with that. Try another spelling.")

with st.expander("How does this work?"):
    st.markdown(
        f"""
1. **Counting.** The app reads {len(BOOKS)} public-domain books and counts every
   sequence of 1 to {MAX_N} words. For example, it counts how often
   *"i do not"* is followed by *"know"*, *"think"*, *"mean"*, …
2. **Predicting.** It takes your last **n − 1** words as the *context* and
   ranks the words that followed that context in the books:
   `score = count(context + word) / count(context)`.
3. **Backing off.** If the context never appeared, it drops the oldest word
   and tries again with a shorter context, multiplying the score by 0.4 each
   time. This is called *stupid backoff* (Brants et al., 2007).
4. **Finishing a word.** If you are in the middle of a word, only words that
   start with those letters are kept.

Try moving the **n** slider in the sidebar to see how more context gives
better guesses.
"""
    )
