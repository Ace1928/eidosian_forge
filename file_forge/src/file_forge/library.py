from __future__ import annotations

import hashlib
import json
import math
import mimetypes
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

ISO_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"
VECTOR_MODEL = "file_forge.deterministic_hash_v1"
VECTOR_DIM = 64
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".rs", ".go", ".java", ".c", ".cpp", ".h", ".hpp", ".sh", ".sql"
}
DOCUMENT_EXTENSIONS = {".md", ".rst", ".txt", ".adoc", ".html", ".htm", ".pdf"}
CONFIG_EXTENSIONS = {".json", ".yaml", ".yml", ".toml", ".ini", ".env", ".conf", ".ndjson", ".code-workspace"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime(ISO_FMT)


def _tokenize(text: str) -> list[str]:
    out: list[str] = []
    token: list[str] = []
    for ch in text.lower():
        if ch.isalnum() or ch in {"_", "-", ".", "/"}:
            token.append(ch)
            continue
        if token:
            out.append("".join(token))
            token = []
    if token:
        out.append("".join(token))
    return out


def build_file_vector(text: str, *, dim: int = VECTOR_DIM) -> tuple[list[float], float]:
    vec = [0.0] * max(8, int(dim))
    for token in _tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % len(vec)
        sign = -1.0 if (digest[4] & 1) else 1.0
        weight = 1.0 + (int.from_bytes(digest[5:7], "big") % 5) / 10.0
        vec[idx] += sign * weight
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0.0:
        vec = [v / norm for v in vec]
    return vec, norm


def file_kind_for_path(path: Path, *, mime_type: Optional[str] = None) -> str:
    suffix = path.suffix.lower()
    if suffix in CODE_EXTENSIONS:
        return "code"
    if suffix in DOCUMENT_EXTENSIONS:
        return "document"
    if suffix in CONFIG_EXTENSIONS:
        return "config"
    if mime_type and mime_type.startswith("text/"):
        return "document"
    if path.name.startswith(".") and not suffix:
        return "config"
    if not suffix:
        return "generic"
    if mime_type and (mime_type.startswith("image/") or mime_type.startswith("audio/") or mime_type.startswith("video/")):
        return "binary"
    return "generic"


def _safe_decode(payload: bytes) -> tuple[Optional[str], Optional[str]]:
    try:
        return payload.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return None, None


def derive_file_links(path: Path, *, kind: str, text_preview: str = "") -> list[dict[str, Any]]:
    links: list[dict[str, Any]] = [{"forge": "file_forge", "relation": "manages", "detail": {"kind": kind}}]
    suffix = path.suffix.lower()
    if kind == "code":
        links.append({"forge": "code_forge", "relation": "captures", "detail": {"language_hint": suffix or None}})
    if kind in {"document", "config", "generic"}:
        links.append({"forge": "knowledge_forge", "relation": "indexes", "detail": {"kind": kind}})
    if kind in {"document", "config", "generic", "code"}:
        links.append({"forge": "word_forge", "relation": "lexicon_seed", "detail": {"kind": kind}})
    lowered = f"{path.as_posix()}\n{text_preview[:512]}".lower()
    if "/doc_forge/runtime/" in lowered or "/final_docs/" in lowered or "/judgments/" in lowered:
        links.append({"forge": "doc_forge", "relation": "documents", "detail": {"kind": kind}})
    if any(token in lowered for token in ("lesson", "memory", "journal", "continuity", "identity", "reflection")):
        links.append({"forge": "memory_forge", "relation": "memory_candidate", "detail": {"signal": "path_or_content"}})
    return links


@dataclass(frozen=True)
class FileRecord:
    file_path: str
    content_hash: str
    size_bytes: int
    modified_ns: int
    kind: str
    mime_type: Optional[str] = None
    encoding: Optional[str] = None
    text_preview: Optional[str] = None


class FileLibraryDB:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS file_blobs (
                    content_hash TEXT PRIMARY KEY,
                    payload BLOB NOT NULL,
                    byte_count INTEGER NOT NULL,
                    encoding TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS file_records (
                    file_path TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    modified_ns INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    mime_type TEXT,
                    encoding TEXT,
                    text_preview TEXT,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS file_vectors (
                    file_path TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    dim INTEGER NOT NULL,
                    vector_json TEXT NOT NULL,
                    norm REAL NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(file_path) REFERENCES file_records(file_path) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS file_links (
                    file_path TEXT NOT NULL,
                    forge_name TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    detail_json TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (file_path, forge_name, relation_type),
                    FOREIGN KEY(file_path) REFERENCES file_records(file_path) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS file_relationships (
                    src_path TEXT NOT NULL,
                    dst_path TEXT NOT NULL,
                    rel_type TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (src_path, dst_path, rel_type)
                );
                """
            )

    def add_blob(self, payload: bytes, *, encoding: Optional[str] = None) -> str:
        content_hash = hashlib.sha256(payload).hexdigest()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO file_blobs (content_hash, payload, byte_count, encoding, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (content_hash, payload, len(payload), encoding, _utc_now()),
            )
        return content_hash

    def get_blob(self, content_hash: str) -> Optional[bytes]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM file_blobs WHERE content_hash = ?",
                (content_hash,),
            ).fetchone()
        if not row:
            return None
        return bytes(row[0])

    def get_file_record(self, file_path: str | Path) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM file_records WHERE file_path = ?",
                (str(Path(file_path).resolve()),),
            ).fetchone()
        return dict(row) if row else None

    def should_process_file(self, file_path: str | Path, *, size_bytes: int, modified_ns: int) -> bool:
        record = self.get_file_record(file_path)
        if not record:
            return True
        return int(record.get("size_bytes") or -1) != int(size_bytes) or int(record.get("modified_ns") or -1) != int(modified_ns)

    def upsert_file(
        self,
        *,
        file_path: str | Path,
        payload: bytes,
        size_bytes: int,
        modified_ns: int,
        kind: str,
        mime_type: Optional[str],
        encoding: Optional[str],
        text_preview: Optional[str],
        links: Iterable[dict[str, Any]],
    ) -> dict[str, Any]:
        path_text = str(Path(file_path).resolve())
        content_hash = self.add_blob(payload, encoding=encoding)
        vector, norm = build_file_vector(f"{Path(path_text).name}\n{kind}\n{text_preview or ''}")
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO file_records (file_path, content_hash, size_bytes, modified_ns, kind, mime_type, encoding, text_preview, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(file_path) DO UPDATE SET
                    content_hash=excluded.content_hash,
                    size_bytes=excluded.size_bytes,
                    modified_ns=excluded.modified_ns,
                    kind=excluded.kind,
                    mime_type=excluded.mime_type,
                    encoding=excluded.encoding,
                    text_preview=excluded.text_preview,
                    updated_at=excluded.updated_at
                """,
                (path_text, content_hash, int(size_bytes), int(modified_ns), kind, mime_type, encoding, text_preview, now),
            )
            conn.execute("DELETE FROM file_links WHERE file_path = ?", (path_text,))
            for link in links:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO file_links (file_path, forge_name, relation_type, detail_json, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        path_text,
                        str(link.get("forge") or "file_forge"),
                        str(link.get("relation") or "manages"),
                        json.dumps(link.get("detail") or {}, sort_keys=True),
                        now,
                    ),
                )
            conn.execute("DELETE FROM file_vectors WHERE file_path = ?", (path_text,))
            conn.execute(
                """
                INSERT INTO file_vectors (file_path, model_name, dim, vector_json, norm, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (path_text, VECTOR_MODEL, len(vector), json.dumps(vector), float(norm), now),
            )
            conn.execute("DELETE FROM file_relationships WHERE src_path = ? OR dst_path = ?", (path_text, path_text))
            parent = str(Path(path_text).resolve().parent)
            conn.execute(
                "INSERT OR REPLACE INTO file_relationships (src_path, dst_path, rel_type, updated_at) VALUES (?, ?, ?, ?)",
                (path_text, parent, "located_in", now),
            )
            if Path(path_text).suffix:
                conn.execute(
                    "INSERT OR REPLACE INTO file_relationships (src_path, dst_path, rel_type, updated_at) VALUES (?, ?, ?, ?)",
                    (path_text, Path(path_text).suffix.lower(), "has_extension", now),
                )
            duplicates = [
                str(row[0])
                for row in conn.execute(
                    "SELECT file_path FROM file_records WHERE content_hash = ? AND file_path != ? ORDER BY file_path ASC",
                    (content_hash, path_text),
                ).fetchall()
            ]
            for other in duplicates:
                conn.execute(
                    "INSERT OR REPLACE INTO file_relationships (src_path, dst_path, rel_type, updated_at) VALUES (?, ?, ?, ?)",
                    (path_text, other, "duplicate_of", now),
                )
                conn.execute(
                    "INSERT OR REPLACE INTO file_relationships (src_path, dst_path, rel_type, updated_at) VALUES (?, ?, ?, ?)",
                    (other, path_text, "duplicate_of", now),
                )
        return {
            "file_path": path_text,
            "content_hash": content_hash,
            "kind": kind,
            "links": list(links),
            "duplicate_count": len(duplicates),
        }

    def iter_file_records(self, *, path_prefix: Optional[str | Path] = None, limit: int = 200000) -> Iterable[dict[str, Any]]:
        where = ""
        params: list[Any] = []
        if path_prefix:
            prefix = str(Path(path_prefix).resolve())
            where = "WHERE file_path = ? OR file_path LIKE ?"
            params.extend([prefix, f"{prefix}/%"])
        params.append(max(1, int(limit)))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM file_records {where} ORDER BY file_path ASC LIMIT ?",
                tuple(params),
            ).fetchall()
        for row in rows:
            yield dict(row)

    def list_links(self, file_path: str | Path) -> list[dict[str, Any]]:
        path_text = str(Path(file_path).resolve())
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT forge_name, relation_type, detail_json FROM file_links WHERE file_path = ? ORDER BY forge_name, relation_type",
                (path_text,),
            ).fetchall()
        out = []
        for row in rows:
            out.append({
                "forge": row[0],
                "relation": row[1],
                "detail": json.loads(row[2]) if row[2] else {},
            })
        return out

    def list_relationships(self, file_path: str | Path) -> list[dict[str, Any]]:
        path_text = str(Path(file_path).resolve())
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT src_path, dst_path, rel_type FROM file_relationships WHERE src_path = ? ORDER BY rel_type, dst_path",
                (path_text,),
            ).fetchall()
        return [{"src_path": row[0], "dst_path": row[1], "rel_type": row[2]} for row in rows]

    def summary(self, *, path_prefix: Optional[str | Path] = None, recent_limit: int = 8) -> dict[str, Any]:
        prefix = str(Path(path_prefix).resolve()) if path_prefix else ""
        recent_limit = max(1, int(recent_limit))
        if prefix:
            record_where = "WHERE file_path = ? OR file_path LIKE ?"
            record_params: list[Any] = [prefix, f"{prefix}/%"]
            link_where = (
                "WHERE file_path IN (SELECT file_path FROM file_records WHERE file_path = ? OR file_path LIKE ?)"
            )
            link_params: list[Any] = [prefix, f"{prefix}/%"]
            rel_where = (
                "WHERE src_path IN (SELECT file_path FROM file_records WHERE file_path = ? OR file_path LIKE ?)"
            )
            rel_params: list[Any] = [prefix, f"{prefix}/%"]
        else:
            record_where = ""
            record_params = []
            link_where = ""
            link_params = []
            rel_where = ""
            rel_params = []

        with self._connect() as conn:
            total_files = int(
                conn.execute(
                    f"SELECT COUNT(*) FROM file_records {record_where}",
                    tuple(record_params),
                ).fetchone()[0]
            )
            total_blobs = int(conn.execute("SELECT COUNT(*) FROM file_blobs").fetchone()[0])
            total_vectors = int(
                conn.execute(
                    f"SELECT COUNT(*) FROM file_vectors WHERE file_path IN (SELECT file_path FROM file_records {record_where})"
                    if record_where
                    else "SELECT COUNT(*) FROM file_vectors",
                    tuple(record_params) if record_where else (),
                ).fetchone()[0]
            )
            total_links = int(
                conn.execute(
                    f"SELECT COUNT(*) FROM file_links {link_where}",
                    tuple(link_params),
                ).fetchone()[0]
            )
            total_relationships = int(
                conn.execute(
                    f"SELECT COUNT(*) FROM file_relationships {rel_where}",
                    tuple(rel_params),
                ).fetchone()[0]
            )
            by_kind_rows = conn.execute(
                f"SELECT kind, COUNT(*) FROM file_records {record_where} GROUP BY kind ORDER BY COUNT(*) DESC, kind ASC",
                tuple(record_params),
            ).fetchall()
            by_forge_rows = conn.execute(
                f"SELECT forge_name, COUNT(*) FROM file_links {link_where} GROUP BY forge_name ORDER BY COUNT(*) DESC, forge_name ASC",
                tuple(link_params),
            ).fetchall()
            recent_rows = conn.execute(
                f"""
                SELECT file_path, kind, updated_at, size_bytes, content_hash
                FROM file_records
                {record_where}
                ORDER BY updated_at DESC, file_path ASC
                LIMIT ?
                """,
                tuple(record_params + [recent_limit]),
            ).fetchall()
            duplicate_groups = int(
                conn.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM (
                        SELECT content_hash
                        FROM file_records
                        {record_where}
                        GROUP BY content_hash
                        HAVING COUNT(*) > 1
                    )
                    """,
                    tuple(record_params),
                ).fetchone()[0]
            )

        return {
            "db_path": str(self.db_path.resolve()),
            "path_prefix": prefix or None,
            "total_files": total_files,
            "total_blobs": total_blobs,
            "total_vectors": total_vectors,
            "total_links": total_links,
            "total_relationships": total_relationships,
            "duplicate_groups": duplicate_groups,
            "by_kind": [{"kind": row[0], "count": int(row[1])} for row in by_kind_rows],
            "by_forge": [{"forge": row[0], "count": int(row[1])} for row in by_forge_rows],
            "recent_files": [
                {
                    "file_path": row[0],
                    "kind": row[1],
                    "updated_at": row[2],
                    "size_bytes": int(row[3]),
                    "content_hash": row[4],
                }
                for row in recent_rows
            ],
        }

    def restore_file(self, *, file_path: str | Path, target_path: str | Path) -> Path:
        record = self.get_file_record(file_path)
        if not record:
            raise FileNotFoundError(f"file record not found: {file_path}")
        payload = self.get_blob(str(record.get("content_hash") or ""))
        if payload is None:
            raise FileNotFoundError(f"content blob not found for: {file_path}")
        target = Path(target_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return target


def index_path(*, db: FileLibraryDB, file_path: Path) -> dict[str, Any]:
    resolved = Path(file_path).resolve()
    stat = resolved.stat()
    modified_ns = int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000)))
    if not db.should_process_file(resolved, size_bytes=int(stat.st_size), modified_ns=modified_ns):
        record = db.get_file_record(resolved) or {}
        return {
            "status": "skipped",
            "file_path": str(resolved),
            "content_hash": record.get("content_hash"),
            "reason": "unchanged",
        }
    payload = resolved.read_bytes()
    text_content, encoding = _safe_decode(payload)
    mime_type, _ = mimetypes.guess_type(str(resolved))
    kind = file_kind_for_path(resolved, mime_type=mime_type)
    preview = (text_content or "")[:4000] if text_content else None
    links = derive_file_links(resolved, kind=kind, text_preview=preview or "")
    result = db.upsert_file(
        file_path=resolved,
        payload=payload,
        size_bytes=int(stat.st_size),
        modified_ns=modified_ns,
        kind=kind,
        mime_type=mime_type,
        encoding=encoding,
        text_preview=preview,
        links=links,
    )
    result["status"] = "indexed"
    return result
