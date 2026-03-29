from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional

from eidosian_core import eidosian
try:
    import morfessor
    MORFESSOR_AVAILABLE = True
except ImportError:
    MORFESSOR_AVAILABLE = False

from word_forge.database.database_manager import DBManager, DatabaseError

LOGGER = logging.getLogger("word_forge.linguistics.morphology")

_COMMON_PREFIXES = (
    "anti", "auto", "hyper", "inter", "micro", "multi", "poly", "post", "pre", "proto", "re", "sub", "trans", "ultra", "un",
)
_COMMON_SUFFIXES = (
    "ization", "isation", "amiento", "imiento", "acion", "ición", "icion", "mente", "ation", "ition", "tion", "ing", "ness", "less", "able", "ible", "ismo", "ista", "idad", "ment", "eur", "euse", "logy",
)

@eidosian()
class MorphologyManager:
    """Manages morphological decomposition and morpheme storage."""

    def __init__(self, db_manager: Optional[DBManager] = None) -> None:
        self.db = db_manager or DBManager()
        self._model = None
        self._model_failed = False
        self._ensure_lexeme_morpheme_tables()
        if MORFESSOR_AVAILABLE:
            self._init_morfessor()

    def _ensure_lexeme_morpheme_tables(self) -> None:
        try:
            with self.db.transaction() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS lexeme_morphemes (
                        lexeme_id INTEGER NOT NULL,
                        morpheme_id INTEGER NOT NULL,
                        position INTEGER NOT NULL,
                        PRIMARY KEY(lexeme_id, position),
                        FOREIGN KEY(lexeme_id) REFERENCES lexemes(id),
                        FOREIGN KEY(morpheme_id) REFERENCES morphemes(id)
                    )
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_lexeme_morphemes_lexeme ON lexeme_morphemes(lexeme_id)")
        except Exception as exc:
            LOGGER.warning("Failed to ensure lexeme morpheme tables: %s", exc)

    def _heuristic_decompose(self, word: str) -> List[str]:
        text = str(word or "").strip().lower()
        if not text:
            return []
        split_parts = [part for part in re.split(r"[_\-\s]+", text) if part]
        if len(split_parts) > 1:
            return split_parts
        for prefix in _COMMON_PREFIXES:
            if text.startswith(prefix) and len(text) - len(prefix) >= 3:
                stem = text[len(prefix):]
                return [prefix, stem]
        for suffix in _COMMON_SUFFIXES:
            if text.endswith(suffix) and len(text) - len(suffix) >= 3:
                stem = text[:-len(suffix)]
                return [stem, suffix]
        return [text]

    def _init_morfessor(self) -> None:
        """Initialize the Morfessor model."""
        try:
            # In a production environment, we'd load a pre-trained model
            # For now, we initialize a baseline model
            self._model = morfessor.BaselineModel()
            LOGGER.info("Morfessor initialized for unsupervised segmentation")
        except Exception as e:
            LOGGER.warning(f"Failed to initialize Morfessor: {e}")

    def decompose(self, word: str) -> List[str]:
        """Decompose a word into its constituent morphemes."""
        if not word:
            return []
        
        if self._model and not self._model_failed:
            try:
                # Basic segmentation using the current model state
                # Note: For better results, the model needs to be trained on a corpus
                return self._model.viterbi_segment(word.lower())[0]
            except Exception as e:
                self._model_failed = True
                LOGGER.warning("Morfessor decomposition unavailable; falling back to heuristic segmentation: %s", e)
        
        return self._heuristic_decompose(word)

    def upsert_morpheme(self, text: str, m_type: Optional[str] = None, meaning: Optional[str] = None) -> int:
        """Insert or update a morpheme record."""
        query = """
        INSERT INTO morphemes (text, type, meaning, last_updated)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(text)
        DO UPDATE SET
            type=COALESCE(excluded.type, type),
            meaning=COALESCE(excluded.meaning, meaning),
            last_updated=excluded.last_updated
        """
        try:
            with self.db.transaction() as conn:
                conn.execute(query, (text.lower().strip(), m_type, meaning, time.time()))
                row = conn.execute(
                    "SELECT id FROM morphemes WHERE text = ?",
                    (text.lower().strip(),),
                ).fetchone()
                if row is None:
                    raise DatabaseError("Failed to retrieve morpheme ID after upsert")
                return int(row[0])
        except Exception as e:
            LOGGER.error(f"Failed to upsert morpheme '{text}': {e}")
            raise

    def set_word_morphemes(self, word_id: int, morpheme_ids: List[int]) -> None:
        """Link a word to its constituent morphemes in order."""
        try:
            with self.db.transaction() as conn:
                conn.execute("DELETE FROM word_morphemes WHERE word_id = ?", (word_id,))
                for pos, m_id in enumerate(morpheme_ids):
                    conn.execute(
                        "INSERT INTO word_morphemes (word_id, morpheme_id, position) VALUES (?, ?, ?)",
                        (word_id, m_id, pos),
                    )
        except Exception as e:
            LOGGER.error(f"Failed to set morphemes for word_id {word_id}: {e}")
            raise

    def set_lexeme_morphemes(self, lexeme_id: int, morpheme_ids: List[int]) -> None:
        """Link a lexeme to its constituent morphemes in order."""
        try:
            with self.db.transaction() as conn:
                conn.execute("DELETE FROM lexeme_morphemes WHERE lexeme_id = ?", (lexeme_id,))
                for pos, m_id in enumerate(morpheme_ids):
                    conn.execute(
                        "INSERT INTO lexeme_morphemes (lexeme_id, morpheme_id, position) VALUES (?, ?, ?)",
                        (lexeme_id, m_id, pos),
                    )
        except Exception as e:
            LOGGER.error(f"Failed to set morphemes for lexeme_id {lexeme_id}: {e}")
            raise

    def get_morphemes(self, word_id: int) -> List[Dict[str, Any]]:
        """Retrieve the ordered list of morphemes for a word."""
        query = """
        SELECT m.*
        FROM morphemes m
        JOIN word_morphemes wm ON m.id = wm.morpheme_id
        WHERE wm.word_id = ?
        ORDER BY wm.position
        """
        try:
            rows = self.db.execute_query(query, (word_id,))
            return [dict(row) for row in rows]
        except Exception as e:
            LOGGER.error(f"Failed to retrieve morphemes for word_id {word_id}: {e}")
            return []

    def get_lexeme_morphemes(self, lexeme_id: int) -> List[Dict[str, Any]]:
        """Retrieve the ordered list of morphemes for a lexeme."""
        query = """
        SELECT m.*
        FROM morphemes m
        JOIN lexeme_morphemes lm ON m.id = lm.morpheme_id
        WHERE lm.lexeme_id = ?
        ORDER BY lm.position
        """
        try:
            rows = self.db.execute_query(query, (lexeme_id,))
            return [dict(row) for row in rows]
        except Exception as e:
            LOGGER.error(f"Failed to retrieve morphemes for lexeme_id {lexeme_id}: {e}")
            return []
