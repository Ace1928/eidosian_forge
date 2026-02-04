"""
Multilingual ingestion and base-language alignment utilities.
"""

from word_forge.multilingual.multilingual_manager import MultilingualManager
from word_forge.multilingual.wiktextract_ingest import (
    ingest_wiktextract_jsonl,
    ingest_kaikki_jsonl,
)

__all__ = [
    "MultilingualManager",
    "ingest_wiktextract_jsonl",
    "ingest_kaikki_jsonl",
]
