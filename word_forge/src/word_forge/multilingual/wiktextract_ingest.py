"""
Ingest multilingual lexicon data from Wiktextract/Kaikki JSONL dumps.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from word_forge.multilingual.multilingual_manager import MultilingualManager

LOGGER = logging.getLogger("word_forge.multilingual.ingest")


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _extract_translations(entry: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    translations = entry.get("translations") or []
    if isinstance(translations, list):
        for t in translations:
            if not isinstance(t, dict):
                continue
            yield {
                "lang": t.get("lang") or t.get("lang_code"),
                "term": t.get("word") or t.get("term") or t.get("translation"),
                "relation": "translation",
            }


def ingest_wiktextract_jsonl(
    path: str,
    manager: Optional[MultilingualManager] = None,
    base_lang: str = "en",
    limit: Optional[int] = None,
) -> None:
    """
    Ingest a Wiktextract JSONL dump.
    """
    manager = manager or MultilingualManager()
    path_obj = Path(path)
    if not path_obj.exists():
        LOGGER.warning("Wiktextract file not found: %s", path)
        return

    count = 0
    for entry in _iter_jsonl(path_obj):
        lemma = str(entry.get("word") or "").strip()
        lang = str(entry.get("lang") or entry.get("lang_code") or "").strip()
        if not lemma or not lang:
            continue
        pos = str(entry.get("pos") or "").strip()
        gloss = ""
        senses = entry.get("senses") or []
        if isinstance(senses, list) and senses:
            sense = senses[0]
            if isinstance(sense, dict):
                glosses = sense.get("glosses") or []
                if glosses:
                    gloss = str(glosses[0])

        lexeme_id = manager.upsert_lexeme(
            lemma=lemma,
            lang=lang,
            part_of_speech=pos,
            gloss=gloss,
            source="wiktextract",
        )

        for translation in _extract_translations(entry):
            target_lang = str(translation.get("lang") or "").strip()
            target_term = str(translation.get("term") or "").strip()
            if not target_lang or not target_term:
                continue
            manager.db.add_translation(
                lexeme_id=lexeme_id,
                target_lang=target_lang,
                target_term=target_term,
                relation="translation",
                source="wiktextract",
            )
            if target_lang == base_lang:
                manager.db.update_lexeme_base(
                    lemma=lemma, lang=lang, base_term=target_term
                )

        count += 1
        if limit and count >= limit:
            break


def ingest_kaikki_jsonl(
    path: str,
    manager: Optional[MultilingualManager] = None,
    base_lang: str = "en",
    limit: Optional[int] = None,
) -> None:
    """
    Ingest a Kaikki.org JSONL dump.
    """
    manager = manager or MultilingualManager()
    path_obj = Path(path)
    if not path_obj.exists():
        LOGGER.warning("Kaikki JSONL file not found: %s", path)
        return

    count = 0
    for entry in _iter_jsonl(path_obj):
        lemma = str(entry.get("word") or "").strip()
        lang = str(entry.get("lang") or entry.get("lang_code") or "").strip()
        if not lemma or not lang:
            continue
        pos = str(entry.get("pos") or "").strip()
        gloss = ""
        senses = entry.get("senses") or []
        if isinstance(senses, list) and senses:
            sense = senses[0]
            if isinstance(sense, dict):
                glosses = sense.get("glosses") or []
                if glosses:
                    gloss = str(glosses[0])

        lexeme_id = manager.upsert_lexeme(
            lemma=lemma,
            lang=lang,
            part_of_speech=pos,
            gloss=gloss,
            source="kaikki",
        )

        translations = entry.get("translations") or []
        if isinstance(translations, list):
            for translation in translations:
                if not isinstance(translation, dict):
                    continue
                target_lang = str(translation.get("lang") or "").strip()
                target_term = str(
                    translation.get("word")
                    or translation.get("term")
                    or translation.get("translation")
                    or ""
                ).strip()
                if not target_lang or not target_term:
                    continue
                manager.db.add_translation(
                    lexeme_id=lexeme_id,
                    target_lang=target_lang,
                    target_term=target_term,
                    relation="translation",
                    source="kaikki",
                )
                if target_lang == base_lang:
                    manager.db.update_lexeme_base(
                        lemma=lemma, lang=lang, base_term=target_term
                    )

        count += 1
        if limit and count >= limit:
            break
