"""
Multilingual manager for Word Forge.

Maintains a base-language (English) alignment layer while ingesting
multilingual lexical resources.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Optional

from word_forge.database.database_manager import DBManager, TermNotFoundError

LOGGER = logging.getLogger("word_forge.multilingual")


class MultilingualManager:
    """Coordinates multilingual ingestion with English as the base language."""

    def __init__(self, db_manager: Optional[DBManager] = None) -> None:
        self.db = db_manager or DBManager()

    def upsert_lexeme(
        self,
        lemma: str,
        lang: str,
        part_of_speech: str = "",
        gloss: str = "",
        source: str = "",
        base_term: str = "",
    ) -> int:
        """Insert/update a lexeme and return its ID."""
        return self.db.insert_or_update_lexeme(
            lemma=lemma,
            lang=lang,
            part_of_speech=part_of_speech,
            gloss=gloss,
            source=source,
            base_term=base_term,
        )

    def add_translation(
        self,
        lemma: str,
        lang: str,
        target_lang: str,
        target_term: str,
        relation: str = "translation",
        source: str = "",
    ) -> None:
        """Add translation mapping for a lexeme."""
        lexeme_id = self.db.get_lexeme_id(lemma=lemma, lang=lang)
        self.db.add_translation(
            lexeme_id=lexeme_id,
            target_lang=target_lang,
            target_term=target_term,
            relation=relation,
            source=source,
        )

    def align_to_english(self, lemma: str, lang: str, english_term: str) -> None:
        """Force-align a lexeme to an English base term."""
        try:
            self.db.update_lexeme_base(lemma=lemma, lang=lang, base_term=english_term)
        except TermNotFoundError:
            self.db.insert_or_update_lexeme(lemma=lemma, lang=lang, base_term=english_term)

    def ingest_translation_batch(
        self,
        lemma: str,
        lang: str,
        translations: Iterable[Dict[str, Any]],
        source: str = "",
    ) -> None:
        """Bulk add translations for a lexeme."""
        for entry in translations:
            target_lang = str(entry.get("lang") or "").strip()
            target_term = str(entry.get("term") or "").strip()
            relation = str(entry.get("relation") or "translation").strip()
            if not target_lang or not target_term:
                continue
            self.add_translation(
                lemma=lemma,
                lang=lang,
                target_lang=target_lang,
                target_term=target_term,
                relation=relation,
                source=source,
            )
