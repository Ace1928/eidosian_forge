from __future__ import annotations

import logging
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

@eidosian()
class MorphologyManager:
    """Manages morphological decomposition and morpheme storage."""

    def __init__(self, db_manager: Optional[DBManager] = None) -> None:
        self.db = db_manager or DBManager()
        self._model = None
        if MORFESSOR_AVAILABLE:
            self._init_morfessor()

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
        
        if self._model:
            try:
                # Basic segmentation using the current model state
                # Note: For better results, the model needs to be trained on a corpus
                return self._model.viterbi_segment(word.lower())[0]
            except Exception as e:
                LOGGER.error(f"Morfessor decomposition failed for '{word}': {e}")
        
        # Simple fallback (this is very basic and should be replaced by model training)
        return [word.lower()]

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
