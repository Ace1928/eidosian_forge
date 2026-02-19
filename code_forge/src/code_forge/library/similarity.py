from __future__ import annotations

import hashlib
import re
from typing import Iterable, Sequence

# Conservative stopword list for code-oriented tokenization.
_STOPWORDS = {
    "the",
    "and",
    "or",
    "to",
    "of",
    "in",
    "for",
    "on",
    "a",
    "an",
    "is",
    "are",
    "be",
    "by",
    "with",
    "as",
    "at",
    "if",
    "else",
    "elif",
    "return",
    "class",
    "def",
    "fn",
    "func",
    "function",
    "public",
    "private",
    "protected",
    "static",
    "const",
    "let",
    "var",
    "true",
    "false",
    "null",
    "none",
    "self",
    "this",
}


def split_identifier(token: str) -> list[str]:
    token = token.strip("_")
    if not token:
        return []
    pieces: list[str] = []
    for part in token.split("_"):
        if not part:
            continue
        camel = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", part)
        pieces.extend(camel.split())
    return [p.lower() for p in pieces if p]


def normalize_code_text(text: str) -> str:
    """Normalize code text into a language-agnostic canonical form."""
    # Strip common single-line comments.
    text = re.sub(r"//.*?$", " ", text, flags=re.MULTILINE)
    text = re.sub(r"#.*?$", " ", text, flags=re.MULTILINE)
    # Strip block comments.
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    # Replace quoted strings with marker token to improve structural matching.
    text = re.sub(r"\"(?:\\.|[^\"\\])*\"", " STR ", text)
    text = re.sub(r"'(?:\\.|[^'\\])*'", " STR ", text)
    # Normalize numeric literals.
    text = re.sub(r"\b\d+(?:\.\d+)?\b", " NUM ", text)
    # Normalize operator spacing so equivalent formatting produces same fingerprint.
    text = re.sub(r"([=+\-*/%<>!&|^~?:,;(){}\[\]])", r" \1 ", text)
    # Canonical whitespace.
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def tokenize_code_text(text: str) -> list[str]:
    normalized = normalize_code_text(text)
    raw = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", normalized)
    out: list[str] = []
    for token in raw:
        for part in split_identifier(token):
            if len(part) <= 1 or part in _STOPWORDS:
                continue
            out.append(part)
    return out


def normalized_hash(text: str) -> str:
    return hashlib.sha256(normalize_code_text(text).encode("utf-8")).hexdigest()


def simhash64(tokens: Sequence[str]) -> int:
    if not tokens:
        return 0
    vector = [0] * 64
    for token in tokens:
        digest = hashlib.sha1(token.encode("utf-8")).digest()[:8]
        value = int.from_bytes(digest, "big", signed=False)
        for bit in range(64):
            if value & (1 << bit):
                vector[bit] += 1
            else:
                vector[bit] -= 1

    out = 0
    for bit, weight in enumerate(vector):
        if weight >= 0:
            out |= (1 << bit)
    return out


def hamming_distance64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def token_jaccard(tokens_a: Iterable[str], tokens_b: Iterable[str]) -> float:
    a = set(tokens_a)
    b = set(tokens_b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a.intersection(b)) / len(a.union(b))


def build_fingerprint(text: str) -> tuple[str, int, int]:
    tokens = tokenize_code_text(text)
    return normalized_hash(text), simhash64(tokens), len(tokens)
