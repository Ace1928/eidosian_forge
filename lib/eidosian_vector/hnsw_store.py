from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None

try:
    import hnswlib
except ImportError as exc:  # pragma: no cover - import validated by callers/tests
    raise RuntimeError("HNSWVectorStore requires hnswlib to be installed.") from exc


@dataclass
class VectorStoreResult:
    item_id: str
    score: float
    text: str
    metadata: Dict[str, Any]


class HNSWVectorStore:
    """Persistent HNSW index with SQLite-backed metadata and POSIX locking."""

    def __init__(
        self,
        persistence_dir: Path | str,
        *,
        dim: Optional[int] = None,
        space: str = "cosine",
        ef_construction: int = 200,
        M: int = 16,
        max_elements: int = 1024,
    ) -> None:
        self.persistence_dir = Path(persistence_dir)
        self.persistence_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite_path = self.persistence_dir / "vectors.sqlite"
        self.index_path = self.persistence_dir / "index.bin"
        self.meta_path = self.persistence_dir / "meta.json"
        self.lock_path = self.persistence_dir / ".vector.lock"
        self.space = str(space)
        self.dim = int(dim) if dim else None
        self.ef_construction = max(16, int(ef_construction))
        self.M = max(4, int(M))
        self.max_elements = max(16, int(max_elements))
        self._thread_lock = threading.RLock()
        self._lock_handle = None
        self._lock_depth = 0
        self._index = None
        self._init_db()
        self._load_meta()
        self._load_index()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.sqlite_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS vector_items (
                    item_id TEXT PRIMARY KEY,
                    label INTEGER NOT NULL UNIQUE,
                    text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    deleted INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_vector_items_label
                    ON vector_items(label);
                CREATE INDEX IF NOT EXISTS idx_vector_items_deleted
                    ON vector_items(deleted);
                """)

    def _load_meta(self) -> None:
        if not self.meta_path.exists():
            return
        try:
            data = json.loads(self.meta_path.read_text(encoding="utf-8"))
        except Exception:
            return
        stored_dim = data.get("dim")
        if self.dim is None and stored_dim is not None:
            self.dim = int(stored_dim)
        self.space = str(data.get("space", self.space))
        self.max_elements = max(self.max_elements, int(data.get("max_elements", self.max_elements)))

    def _save_meta(self) -> None:
        self.meta_path.write_text(
            json.dumps(
                {
                    "dim": self.dim,
                    "space": self.space,
                    "max_elements": self.max_elements,
                    "ef_construction": self.ef_construction,
                    "M": self.M,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _load_index(self) -> None:
        if self.dim is None:
            return
        self._index = hnswlib.Index(space=self.space, dim=self.dim)
        if self.index_path.exists():
            self._index.load_index(str(self.index_path), max_elements=self.max_elements, allow_replace_deleted=True)
        else:
            self._index.init_index(
                max_elements=self.max_elements,
                ef_construction=self.ef_construction,
                M=self.M,
                allow_replace_deleted=True,
            )
        self._index.set_ef(max(50, min(self.max_elements, 200)))

    def _ensure_index(self, vector: Iterable[float]) -> list[float]:
        dense = [float(v) for v in vector]
        if not dense:
            raise ValueError("Vector cannot be empty")
        if self.dim is None:
            self.dim = len(dense)
            self._load_index()
            self._save_meta()
        elif len(dense) != self.dim:
            raise ValueError(f"Vector dimension mismatch: expected {self.dim}, got {len(dense)}")
        return dense

    def _active_count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM vector_items WHERE deleted = 0").fetchone()
        return int(row["c"]) if row else 0

    def _next_label(self, conn: sqlite3.Connection) -> int:
        row = conn.execute("SELECT COALESCE(MAX(label), -1) AS max_label FROM vector_items").fetchone()
        max_label = -1 if row is None or row["max_label"] is None else int(row["max_label"])
        return max_label + 1

    def _ensure_capacity(self, target_count: int) -> None:
        if self._index is None:
            return
        if target_count <= self.max_elements:
            return
        while self.max_elements < target_count:
            self.max_elements *= 2
        self._index.resize_index(self.max_elements)
        self._save_meta()

    def upsert(
        self,
        item_id: str,
        vector: Iterable[float],
        *,
        text: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        dense = self._ensure_index(vector)
        payload = json.dumps(metadata or {}, sort_keys=True)
        with self._lock():
            if self._index is None:
                raise RuntimeError("Vector index is not initialized")
            vec = np.array([dense], dtype=np.float32)
            with self._connect() as conn:
                existing = conn.execute(
                    "SELECT label, deleted FROM vector_items WHERE item_id = ?",
                    (str(item_id),),
                ).fetchone()
                if existing:
                    label = int(existing["label"])
                    deleted = bool(existing["deleted"])
                else:
                    label = self._next_label(conn)
                    deleted = False
                self._ensure_capacity(max(self._active_count() + (0 if existing and not deleted else 1), label + 1))
                labels = np.array([label], dtype=np.int64)
                try:
                    self._index.add_items(vec, labels, replace_deleted=bool(existing and deleted))
                except TypeError:
                    self._index.add_items(vec, labels)
                conn.execute(
                    """
                    INSERT INTO vector_items (item_id, label, text, metadata_json, deleted, updated_at)
                    VALUES (?, ?, ?, ?, 0, CURRENT_TIMESTAMP)
                    ON CONFLICT(item_id) DO UPDATE SET
                        label=excluded.label,
                        text=excluded.text,
                        metadata_json=excluded.metadata_json,
                        deleted=0,
                        updated_at=CURRENT_TIMESTAMP
                    """,
                    (str(item_id), label, str(text), payload),
                )
            self._index.save_index(str(self.index_path))
            self._save_meta()

    def delete(self, item_id: str) -> bool:
        with self._lock():
            if self._index is None:
                return False
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT label, deleted FROM vector_items WHERE item_id = ?",
                    (str(item_id),),
                ).fetchone()
                if row is None or bool(row["deleted"]):
                    return False
                label = int(row["label"])
                self._index.mark_deleted(label)
                conn.execute(
                    "UPDATE vector_items SET deleted = 1, updated_at = CURRENT_TIMESTAMP WHERE item_id = ?",
                    (str(item_id),),
                )
            self._index.save_index(str(self.index_path))
            return True

    def count(self) -> int:
        return self._active_count()

    def query(
        self,
        vector: Iterable[float],
        *,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        overfetch: Optional[int] = None,
    ) -> list[VectorStoreResult]:
        dense = self._ensure_index(vector)
        if self._index is None or self._active_count() == 0:
            return []
        query_vec = np.array([dense], dtype=np.float32)
        k = min(max(1, int(overfetch or max(limit * 8, 32))), self._active_count())
        labels, distances = self._index.knn_query(query_vec, k=k)
        label_ids = [int(label) for label in labels[0] if int(label) >= 0]
        if not label_ids:
            return []
        placeholders = ",".join(["?"] * len(label_ids))
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT item_id, label, text, metadata_json
                FROM vector_items
                WHERE deleted = 0 AND label IN ({placeholders})
                """,
                tuple(label_ids),
            ).fetchall()
        by_label = {int(row["label"]): row for row in rows}
        results: list[VectorStoreResult] = []
        for label, distance in zip(labels[0], distances[0]):
            row = by_label.get(int(label))
            if row is None:
                continue
            metadata = json.loads(str(row["metadata_json"] or "{}"))
            if not self._matches_filters(metadata, filters):
                continue
            score = 1.0 - float(distance) if self.space == "cosine" else -float(distance)
            results.append(
                VectorStoreResult(
                    item_id=str(row["item_id"]),
                    score=score,
                    text=str(row["text"] or ""),
                    metadata=metadata,
                )
            )
            if len(results) >= max(1, int(limit)):
                break
        return results

    def _matches_filters(self, metadata: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
        if not filters:
            return True
        for key, expected in filters.items():
            actual = metadata.get(key)
            if isinstance(expected, (list, tuple, set)):
                values = set(expected)
                if isinstance(actual, list):
                    if not values.intersection(actual):
                        return False
                elif actual not in values:
                    return False
            elif isinstance(actual, list):
                if expected not in actual:
                    return False
            elif actual != expected:
                return False
        return True

    @contextmanager
    def _lock(self):
        with self._thread_lock:
            if self._lock_depth == 0:
                self._lock_handle = open(self.lock_path, "a+", encoding="utf-8")
                if fcntl is not None:
                    fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_EX)
            self._lock_depth += 1
        try:
            yield
        finally:
            with self._thread_lock:
                self._lock_depth -= 1
                if self._lock_depth == 0 and self._lock_handle is not None:
                    if fcntl is not None:
                        fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
                    self._lock_handle.close()
                    self._lock_handle = None
