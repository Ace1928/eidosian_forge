from eidosian_core import eidosian
"""
Response Caching.
"""
import sqlite3
import json
from pathlib import Path
from typing import Optional
from ..core.interfaces import LLMResponse

class SQLiteCache:
    def __init__(self, db_path: str = "./data/llm_cache.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS responses (
                    key TEXT PRIMARY KEY,
                    response TEXT,
                    model TEXT,
                    tokens INTEGER
                )
            """)

    @eidosian()
    def get(self, key: str) -> Optional[LLMResponse]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT response, model, tokens FROM responses WHERE key = ?", (key,))
            row = cursor.fetchone()
            if row:
                return LLMResponse(
                    text=row[0],
                    model_name=row[1],
                    tokens_used=row[2],
                    meta={"cached": True}
                )
        return None

    @eidosian()
    def set(self, key: str, response: LLMResponse):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO responses (key, response, model, tokens) VALUES (?, ?, ?, ?)",
                (key, response.text, response.model_name, response.tokens_used)
            )
