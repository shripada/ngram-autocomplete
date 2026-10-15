"""Download a small public-domain corpus from Project Gutenberg and cache it locally."""

from pathlib import Path
import urllib.request

DATA_DIR = Path(__file__).parent / "data"

# Gutenberg ebook id -> title. All public domain.
BOOKS = {
    1342: "Pride and Prejudice",
    1661: "The Adventures of Sherlock Holmes",
    11: "Alice's Adventures in Wonderland",
    84: "Frankenstein",
    98: "A Tale of Two Cities",
    1400: "Great Expectations",
    345: "Dracula",
    2701: "Moby Dick",
}


def _strip_gutenberg_boilerplate(text: str) -> str:
    start = text.find("*** START OF")
    end = text.find("*** END OF")
    if start != -1:
        start = text.find("\n", start) + 1
    else:
        start = 0
    if end == -1:
        end = len(text)
    return text[start:end]


def load_book(book_id: int) -> str:
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"pg{book_id}.txt"
    if not path.exists():
        url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        print(f"Downloading {url}")
        with urllib.request.urlopen(url, timeout=60) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        path.write_text(_strip_gutenberg_boilerplate(raw), encoding="utf-8")
    return path.read_text(encoding="utf-8")


def load_corpus(book_ids) -> str:
    return "\n".join(load_book(b) for b in book_ids)
