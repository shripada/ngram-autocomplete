"""A word-level n-gram language model with "stupid backoff".

How it predicts the next word:
  1. Take the last (n-1) words you typed as the context.
  2. Look up every word that followed that context in the corpus, and how often.
  3. If the context was never seen, drop the oldest word and try again
     (4-gram -> trigram -> bigram -> unigram). Each step back multiplies
     the score by ALPHA, so longer matches are preferred.
"""

from collections import Counter, defaultdict
import re

BOS = "<s>"  # start of sentence
ALPHA = 0.4  # backoff penalty from Brants et al. (2007), "stupid backoff"
TOP_K_PER_CONTEXT = 25  # keep only the most frequent followers to save memory

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n\s*\n")
_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower().replace("’", "'"))


def sentences(text: str):
    for sent in _SENT_SPLIT.split(text):
        words = tokenize(sent)
        if words:
            yield words


class NGramModel:
    def __init__(self, n: int = 4):
        self.n = n
        # followers[k][context_tuple] -> list of (word, count), most frequent first.
        # k is the context length: 0 = unigram, 1 = bigram, ...
        self.followers: list[dict[tuple, list[tuple[str, int]]]] = []
        self.context_totals: list[dict[tuple, int]] = []
        self.num_tokens = 0

    def fit(self, text: str) -> "NGramModel":
        counts = [defaultdict(Counter) for _ in range(self.n)]
        for words in sentences(text):
            padded = [BOS] * (self.n - 1) + words
            self.num_tokens += len(words)
            for i in range(self.n - 1, len(padded)):
                word = padded[i]
                for k in range(self.n):
                    ctx = tuple(padded[i - k : i])
                    counts[k][ctx][word] += 1

        for k in range(self.n):
            followers, totals = {}, {}
            for ctx, counter in counts[k].items():
                totals[ctx] = sum(counter.values())
                followers[ctx] = counter.most_common(TOP_K_PER_CONTEXT if k else None)
            self.followers.append(followers)
            self.context_totals.append(totals)
        return self

    def predict(self, text: str, k: int = 5, order: int | None = None):
        """Suggest the next word (or finish the word being typed).

        Returns (suggestions, info). Each suggestion is
        (word, score, context_found, times_seen): context_found is the run of
        words that was actually followed by `word` in the corpus.
        """
        order = min(order or self.n, self.n)
        ends_mid_word = bool(text) and not text[-1].isspace() and text[-1].isalpha()
        # Only the current sentence counts as context.
        last_stop = max(text.rfind(c) for c in ".!?")
        history = tokenize(text[last_stop + 1 :])
        prefix = history.pop() if ends_mid_word and history else ""

        padded = [BOS] * (self.n - 1) + history
        context = tuple(padded[len(padded) - (order - 1) :]) if order > 1 else ()

        best: dict[str, tuple[float, tuple, int]] = {}
        weight = 1.0
        for ctx_len in range(len(context), -1, -1):
            ctx = context[len(context) - ctx_len :]
            followers = self.followers[ctx_len].get(ctx)
            if followers:
                total = self.context_totals[ctx_len][ctx]
                for word, count in followers:
                    if prefix and not word.startswith(prefix):
                        continue
                    if word == prefix:
                        continue
                    score = weight * count / total
                    if word not in best or score > best[word][0]:
                        best[word] = (score, ctx, count)
            if len(best) >= k:
                break
            weight *= ALPHA

        ranked = sorted(best.items(), key=lambda kv: -kv[1][0])[:k]
        suggestions = [(w, s, c, n) for w, (s, c, n) in ranked]
        info = {"context": list(context), "prefix": prefix}
        return suggestions, info
