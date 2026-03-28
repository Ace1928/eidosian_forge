from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from word_forge.database.database_manager import DBManager
from word_forge.multilingual import ingest_kaikki_jsonl, ingest_wiktextract_jsonl
from word_forge.multilingual.multilingual_manager import MultilingualManager


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def append_history(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def db_counts(db_path: Path) -> dict[str, Any]:
    if not db_path.exists():
        return {
            "lexeme_count": 0,
            "translation_count": 0,
            "base_aligned_count": 0,
            "language_count": 0,
            "top_languages": [],
            "top_translation_languages": [],
        }
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        lexeme_count = int(conn.execute("SELECT COUNT(*) FROM lexemes").fetchone()[0])
        translation_count = int(conn.execute("SELECT COUNT(*) FROM translations").fetchone()[0])
        base_aligned_count = int(
            conn.execute("SELECT COUNT(*) FROM lexemes WHERE COALESCE(base_term, '') != ''").fetchone()[0]
        )
        languages = conn.execute(
            "SELECT lang, COUNT(*) AS count FROM lexemes GROUP BY lang ORDER BY count DESC LIMIT 12"
        ).fetchall()
        translation_languages = conn.execute(
            "SELECT target_lang AS lang, COUNT(*) AS count FROM translations GROUP BY target_lang ORDER BY count DESC LIMIT 12"
        ).fetchall()
    return {
        "lexeme_count": lexeme_count,
        "translation_count": translation_count,
        "base_aligned_count": base_aligned_count,
        "language_count": len(languages),
        "top_languages": [{"lang": row["lang"], "count": int(row["count"])} for row in languages],
        "top_translation_languages": [{"lang": row["lang"], "count": int(row["count"])} for row in translation_languages],
    }


def build_multilingual_ingest_report(
    source_path: Path,
    source_type: str,
    db_path: Path,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    manager = MultilingualManager(DBManager(db_path=db_path))
    before = db_counts(db_path)
    if source_type == "kaikki":
        ingest_kaikki_jsonl(str(source_path), manager=manager, limit=limit)
    elif source_type == "wiktextract":
        ingest_wiktextract_jsonl(str(source_path), manager=manager, limit=limit)
    else:
        raise ValueError(f"Unsupported source_type: {source_type}")
    after = db_counts(db_path)
    deltas = {
        "lexeme_delta": after["lexeme_count"] - before["lexeme_count"],
        "translation_delta": after["translation_count"] - before["translation_count"],
        "base_aligned_delta": after["base_aligned_count"] - before["base_aligned_count"],
    }
    return {
        "contract": "eidos.word_forge.multilingual_ingest.v1",
        "generated_at": now_iso(),
        "source_type": source_type,
        "source_path": str(source_path),
        "source_signature": source_signature(source_path),
        "db_path": str(db_path),
        "limit": limit,
        "before": before,
        "after": after,
        "deltas": deltas,
    }


def render_multilingual_ingest_markdown(report: dict[str, Any]) -> str:
    after = report.get("after") or {}
    deltas = report.get("deltas") or {}
    top_languages = after.get("top_languages") or []
    top_translation_languages = after.get("top_translation_languages") or []
    lines = [
        "# Word Forge Multilingual Ingest",
        "",
        f"- Generated: `{report.get('generated_at')}`",
        f"- Source type: `{report.get('source_type')}`",
        f"- Source path: `{report.get('source_path')}`",
        f"- Lexeme delta: `{deltas.get('lexeme_delta', 0)}`",
        f"- Translation delta: `{deltas.get('translation_delta', 0)}`",
        f"- Base aligned delta: `{deltas.get('base_aligned_delta', 0)}`",
        "",
        "## Totals",
        "",
        f"- Lexemes: `{after.get('lexeme_count', 0)}`",
        f"- Translations: `{after.get('translation_count', 0)}`",
        f"- Base aligned: `{after.get('base_aligned_count', 0)}`",
        f"- Languages: `{after.get('language_count', 0)}`",
        "",
        "## Top Lexeme Languages",
        "",
    ]
    lines.extend(
        [f"- `{row.get('lang')}`: `{row.get('count')}`" for row in top_languages] or ["- None"]
    )
    lines.extend(["", "## Top Translation Languages", ""])
    lines.extend(
        [f"- `{row.get('lang')}`: `{row.get('count')}`" for row in top_translation_languages] or ["- None"]
    )
    return "\n".join(lines) + "\n"


def write_multilingual_ingest_report(repo_root: Path, report: dict[str, Any]) -> dict[str, str]:
    report_dir = repo_root / "reports" / "word_forge_multilingual_ingest"
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = report_dir / f"word_forge_multilingual_ingest_{stamp}.json"
    md_path = report_dir / f"word_forge_multilingual_ingest_{stamp}.md"
    latest_json = report_dir / "latest.json"
    latest_md = report_dir / "latest.md"
    write_json(json_path, report)
    md_path.write_text(render_multilingual_ingest_markdown(report), encoding="utf-8")
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_md.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    return {
        "json_path": str(json_path),
        "markdown_path": str(md_path),
        "latest_json": str(latest_json),
        "latest_markdown": str(latest_md),
    }


def run_multilingual_ingest(
    repo_root: Path,
    source_path: Path,
    source_type: str,
    db_path: Path,
    *,
    limit: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    runtime_dir = repo_root / "data" / "runtime"
    status_path = runtime_dir / "word_forge_multilingual_ingest_status.json"
    history_path = runtime_dir / "word_forge_multilingual_ingest_history.jsonl"
    signature = source_signature(source_path)
    previous = read_json(status_path)
    status_payload = {
        "contract": "eidos.word_forge.multilingual_ingest.status.v1",
        "status": "running",
        "phase": "ingesting",
        "source_type": source_type,
        "source_path": str(source_path),
        "source_signature": signature,
        "db_path": str(db_path),
        "started_at": now_iso(),
        "limit": limit,
        "force": force,
    }
    write_json(status_path, status_payload)
    try:
        if not force and previous.get("source_signature") == signature and previous.get("status") == "completed":
            status_payload.update({"status": "skipped", "phase": "unchanged", "finished_at": now_iso()})
            write_json(status_path, status_payload)
            append_history(history_path, status_payload)
            return {"status": status_payload, "report": None, "artifacts": {}}

        report = build_multilingual_ingest_report(source_path, source_type, db_path, limit=limit)
        artifacts = write_multilingual_ingest_report(repo_root, report)
        status_payload.update(
            {
                "status": "completed",
                "phase": "completed",
                "finished_at": now_iso(),
                "report_path": artifacts["latest_json"],
                "lexeme_delta": report["deltas"]["lexeme_delta"],
                "translation_delta": report["deltas"]["translation_delta"],
                "base_aligned_delta": report["deltas"]["base_aligned_delta"],
            }
        )
        write_json(status_path, status_payload)
        append_history(history_path, status_payload)
        return {"status": status_payload, "report": report, "artifacts": artifacts}
    except Exception as exc:
        status_payload.update({
            "status": "failed",
            "phase": "failed",
            "finished_at": now_iso(),
            "error": str(exc),
        })
        write_json(status_path, status_payload)
        append_history(history_path, status_payload)
        raise
