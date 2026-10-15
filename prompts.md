# How this app was built

This app was built in one conversation with Claude Code (Claude Opus 5.5). Below are the
prompts, in order, and what each one led to.

## 1. Build the app

> Can you build a streamlit program that uses n grams under the hood to predict next token
> when user types in a text field, options will be shown in a popup there. build it outside
> of this project, use uv package manager. Think of using an open source corpus

What was decided and built:

- **Project:** a new uv project, `uv init --app`, with `streamlit` and `streamlit-searchbox`.
- **Popup:** plain Streamlit has no text box that shows suggestions as you type, so the app
  uses [`streamlit-searchbox`](https://github.com/m-wrzr/streamlit-searchbox). With
  `edit_after_submit="option"`, picking a suggestion puts the whole text back in the box,
  so you can keep typing.
- **Corpus:** 8 public-domain books from Project Gutenberg, about 1 million words
  (`corpus.py`). They download once into `data/`, and the Gutenberg header and footer are
  removed.
- **Model** (`ngram.py`):
  - Word n-grams up to 4 words, counted per sentence, with sentence-start padding.
  - Ranking uses "stupid backoff": `count(context + word) / count(context)`. If the context
    was never seen, the oldest word is dropped and the score is multiplied by 0.4.
  - If you stop partway through a word, only words starting with those letters are suggested.
  - Each context keeps only its 25 most common next words, to save memory.
- **Testing:** the model was tried on sample phrases ("i do not" → know, think, mean;
  "Sherlock" → holmes), and the app was checked in the browser.
- **Bug fixed:** when you stopped mid-word, the context lost a word.
- **Bug fixed:** the empty-text suggestions overwrote the explanation panel's state on
  every rerun.

## 2. Make the explanation clearer

> guess "What the model sees" section is very confusing there.

The old panel showed big number boxes, raw scores and labels like "4-gram". It was replaced
by **"Why these suggestions?"**:

- **One sentence** says which words the model looked at, and which partial word it is
  finishing.
- **A table** lists each suggestion, the exact phrase found in the books, and how many times
  it was seen. For example, "do not know **what**" was seen 14 times.
- **A note** appears only when the model had to fall back to a shorter phrase (try "the
  purple elephant").

To make this possible, `NGramModel.predict` now returns the context it actually matched and
the count, instead of the n-gram order.

## 3. Publish it

> push it to a git remote public repo and link it in slides of w03 and publish

- **Cleanup:** the unused `src/` package that `uv init` had created was removed.
- **Repo:** pushed to this public repo.
- **Slide:** added a slide, "The same counting, on 8 books", to Workshop 3 of the ADS 6703
  NLP course, linking here, with a demo script in the speaker notes.

## 4. This file

> Can you summarise this conversation we used to build this app in a markdown called
> prompts.md and push it to that project repo?

## Run it

```bash
uv sync
uv run streamlit run app.py
```

Things to try: "I do not", "Sherlock", "the purple elephant". Then move the **n** slider down
to 2 and 1 to see how the guesses get worse with less context.
