from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Tuple

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

try:
    from docx import Document as DocxDocument
except Exception:
    DocxDocument = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

from .config import ScribeConfig

COMMON_STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "this",
    "with",
    "from",
    "into",
    "your",
    "file",
    "code",
    "data",
    "are",
    "was",
    "were",
    "has",
    "have",
    "had",
    "will",
    "would",
    "should",
    "could",
    "can",
    "not",
    "but",
    "you",
    "our",
    "their",
    "its",
    "then",
    "than",
    "when",
    "where",
    "what",
    "which",
    "there",
    "here",
    "about",
    "over",
    "under",
    "also",
    "while",
    "using",
    "used",
    "use",
    "each",
    "per",
    "all",
    "any",
    "may",
    "one",
    "two",
    "three",
    "four",
    "five",
    "via",
    "out",
    "in",
    "on",
    "as",
    "at",
    "to",
    "of",
    "is",
    "it",
    "by",
    "an",
    "or",
    "be",
    "if",
    "do",
    "does",
}


def is_text_bytes(sample: bytes) -> bool:
    if not sample:
        return True
    if b"\x00" in sample:
        return False
    text_bytes = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)))
    return all(ch in text_bytes for ch in sample)


def extract_terms(text: str, limit: int = 24) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_\-]{3,}", text.lower())
    words = [w for w in words if w not in COMMON_STOPWORDS]
    freq = Counter(words)
    return [w for w, _ in freq.most_common(limit)]


def extract_symbols(source_text: str) -> list[str]:
    patterns = [
        r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)",
        r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)",
        r"\bfunction\s+([A-Za-z_][A-Za-z0-9_]*)",
        r"\bconst\s+([A-Za-z_][A-Za-z0-9_]*)",
        r"\blet\s+([A-Za-z_][A-Za-z0-9_]*)",
        r"\binterface\s+([A-Za-z_][A-Za-z0-9_]*)",
        r"\benum\s+([A-Za-z_][A-Za-z0-9_]*)",
    ]
    out: list[str] = []
    for pattern in patterns:
        out.extend(re.findall(pattern, source_text))
    unique: list[str] = []
    seen: set[str] = set()
    for item in out:
        if item not in seen:
            unique.append(item)
            seen.add(item)
    return unique[:80]


class DocumentExtractor:
    def __init__(self, cfg: ScribeConfig) -> None:
        self.cfg = cfg

    def extract(self, path: Path) -> Tuple[str, dict[str, Any]]:
        suffix = path.suffix.lower()
        if suffix in {".pdf"}:
            return self._extract_pdf(path), {"doc_type": "pdf"}
        if suffix in {".docx"}:
            return self._extract_docx(path), {"doc_type": "docx"}
        if suffix in {".html", ".htm", ".xhtml", ".svg", ".xml"}:
            return self._extract_html_like(path), {"doc_type": suffix.lstrip(".")}
        return self._extract_text(path), {"doc_type": suffix.lstrip(".") or "text"}

    def _extract_text(self, path: Path) -> str:
        raw = path.read_bytes()
        if len(raw) > self.cfg.max_file_bytes:
            raw = raw[: self.cfg.max_file_bytes]
        if not is_text_bytes(raw[:4096]):
            raise ValueError("binary-like content")
        text = raw.decode("utf-8", errors="replace")
        return text[: self.cfg.max_chars]

    def _extract_pdf(self, path: Path) -> str:
        if PdfReader is None:
            raise RuntimeError("pypdf unavailable")
        reader = PdfReader(str(path))
        parts: list[str] = []
        for page in reader.pages:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                continue
            if sum(len(p) for p in parts) > self.cfg.max_chars:
                break
        text = "\n\n".join(parts)
        return text[: self.cfg.max_chars]

    def _extract_docx(self, path: Path) -> str:
        if DocxDocument is None:
            raise RuntimeError("python-docx unavailable")
        doc = DocxDocument(str(path))
        text = "\n".join(p.text for p in doc.paragraphs)
        return text[: self.cfg.max_chars]

    def _extract_html_like(self, path: Path) -> str:
        text = path.read_text(encoding="utf-8", errors="replace")
        if BeautifulSoup is None:
            return text[: self.cfg.max_chars]
        soup = BeautifulSoup(text, "html.parser")
        body_text = soup.get_text("\n", strip=True)
        return body_text[: self.cfg.max_chars]
