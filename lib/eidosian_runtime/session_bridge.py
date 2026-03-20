from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

FORGE_ROOT = Path(__file__).resolve().parents[2]
HOME_ROOT = FORGE_ROOT.parent
DEFAULT_SESSION_RUNTIME = FORGE_ROOT / 'data' / 'runtime' / 'session_bridge'
DEFAULT_EVENTS_PATH = DEFAULT_SESSION_RUNTIME / 'events.jsonl'
DEFAULT_STATUS_PATH = DEFAULT_SESSION_RUNTIME / 'latest_context.json'
DEFAULT_IMPORT_STATUS_PATH = DEFAULT_SESSION_RUNTIME / 'import_status.json'


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(value: Any) -> datetime | None:
    text = str(value or '').strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace('Z', '+00:00'))
    except Exception:
        return None


def new_session_id(interface: str) -> str:
    return f"{interface}:{uuid.uuid4().hex[:12]}"


def _safe_json_loads(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _load_import_status(path: str | Path | None = None) -> dict[str, Any]:
    target = Path(path or DEFAULT_IMPORT_STATUS_PATH)
    if not target.exists():
        return {'gemini': {}, 'codex': {}}
    payload = _safe_json_loads(target.read_text(encoding='utf-8'))
    if payload is None:
        return {'gemini': {}, 'codex': {}}
    payload.setdefault('gemini', {})
    payload.setdefault('codex', {})
    return payload


def _write_import_status(payload: dict[str, Any], path: str | Path | None = None) -> None:
    target = Path(path or DEFAULT_IMPORT_STATUS_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + '\n', encoding='utf-8')


def append_session_event(
    *,
    interface: str,
    session_id: str,
    event_type: str,
    summary: str,
    metadata: dict[str, Any] | None = None,
    events_path: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(events_path or DEFAULT_EVENTS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'contract': 'eidos.session_event.v1',
        'ts': _now_iso(),
        'interface': str(interface),
        'session_id': str(session_id),
        'event_type': str(event_type),
        'summary': str(summary).strip(),
        'metadata': dict(metadata or {}),
    }
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + '\n')
    return payload


def import_gemini_journal(
    *,
    home_root: str | Path | None = None,
    events_path: str | Path | None = None,
    import_status_path: str | Path | None = None,
    limit: int = 80,
) -> dict[str, Any]:
    home = Path(home_root or HOME_ROOT)
    journal = home / '.gemini' / 'context_memory' / 'user.journal.jsonl'
    if not journal.exists():
        return {'source': 'gemini', 'imported': 0, 'status': 'missing'}
    status = _load_import_status(import_status_path)
    imported_ids = {str(x) for x in status.get('gemini', {}).get('imported_ids', [])}
    lines = journal.read_text(encoding='utf-8', errors='ignore').splitlines()
    imported = 0
    for line in lines[-max(1, int(limit)):]:
        payload = _safe_json_loads(line)
        if payload is None:
            continue
        row_id = str(payload.get('id') or '')
        text = str(payload.get('text') or '').strip()
        if not row_id or not text or row_id in imported_ids:
            continue
        append_session_event(
            interface='gemini',
            session_id='gemini:journal',
            event_type='memory_import',
            summary=text[:200],
            metadata={
                'source': 'gemini_journal',
                'journal_id': row_id,
                'scope': str(payload.get('scope') or ''),
                'op': str(payload.get('op') or ''),
            },
            events_path=events_path,
        )
        imported_ids.add(row_id)
        imported += 1
    status['gemini']['imported_ids'] = sorted(imported_ids)[-2000:]
    status['gemini']['last_imported_at'] = _now_iso()
    _write_import_status(status, import_status_path)
    return {'source': 'gemini', 'imported': imported, 'status': 'ok'}


def _extract_codex_summary(payload: dict[str, Any]) -> tuple[str, str] | None:
    row_type = str(payload.get('type') or '')
    body = payload.get('payload') if isinstance(payload.get('payload'), dict) else {}
    if row_type == 'event_msg' and str(body.get('type') or '') == 'user_message':
        message = str(body.get('message') or '').strip()
        return ('user', message[:200]) if message else None
    if row_type != 'response_item':
        return None
    if str(body.get('type') or '') != 'message':
        return None
    role = str(body.get('role') or '').strip()
    items = body.get('content') if isinstance(body.get('content'), list) else []
    texts = [
        str(item.get('text') or '').strip()
        for item in items
        if isinstance(item, dict) and str(item.get('type') or '') == 'input_text' and str(item.get('text') or '').strip()
    ]
    if role not in {'user', 'assistant'} or not texts:
        return None
    return (role, ' '.join(texts)[:200])


def import_codex_rollouts(
    *,
    home_root: str | Path | None = None,
    events_path: str | Path | None = None,
    import_status_path: str | Path | None = None,
    thread_limit: int = 3,
    events_per_thread: int = 8,
) -> dict[str, Any]:
    home = Path(home_root or HOME_ROOT)
    db_path = home / '.codex' / 'state_5.sqlite'
    if not db_path.exists():
        return {'source': 'codex', 'imported': 0, 'status': 'missing'}
    status = _load_import_status(import_status_path)
    imported_threads = status.get('codex', {}).get('threads', {})
    imported = 0
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            'select id, rollout_path, updated_at from threads order by updated_at desc limit ?',
            (max(1, int(thread_limit)),),
        ).fetchall()
    finally:
        conn.close()
    for thread_id, rollout_path, updated_at in rows:
        thread_key = str(thread_id)
        if imported_threads.get(thread_key) == int(updated_at):
            continue
        path = Path(str(rollout_path))
        if not path.exists():
            continue
        session_id = f"codex:{thread_key[:12]}"
        messages: list[tuple[str, str]] = []
        for line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
            payload = _safe_json_loads(line)
            if payload is None:
                continue
            extracted = _extract_codex_summary(payload)
            if extracted is not None:
                messages.append(extracted)
        for role, summary in messages[-max(1, int(events_per_thread)):]:
            append_session_event(
                interface='codex',
                session_id=session_id,
                event_type=f'{role}_import',
                summary=summary,
                metadata={'source': 'codex_rollout', 'thread_id': thread_key},
                events_path=events_path,
            )
            imported += 1
        imported_threads[thread_key] = int(updated_at)
    status['codex']['threads'] = imported_threads
    status['codex']['last_imported_at'] = _now_iso()
    _write_import_status(status, import_status_path)
    return {'source': 'codex', 'imported': imported, 'status': 'ok'}


def sync_external_sessions(
    *,
    home_root: str | Path | None = None,
    events_path: str | Path | None = None,
    import_status_path: str | Path | None = None,
    min_interval_sec: float = 60.0,
) -> dict[str, Any]:
    status = _load_import_status(import_status_path)
    last_sync = _parse_iso(status.get('last_sync_at'))
    if last_sync is not None:
        age = (datetime.now(timezone.utc) - last_sync).total_seconds()
        if age < max(1.0, float(min_interval_sec)):
            return {'gemini': {'status': 'skipped_recent'}, 'codex': {'status': 'skipped_recent'}}
    gemini = import_gemini_journal(
        home_root=home_root,
        events_path=events_path,
        import_status_path=import_status_path,
    )
    codex = import_codex_rollouts(
        home_root=home_root,
        events_path=events_path,
        import_status_path=import_status_path,
    )
    status = _load_import_status(import_status_path)
    status['last_sync_at'] = _now_iso()
    _write_import_status(status, import_status_path)
    return {'gemini': gemini, 'codex': codex}


def read_session_events(limit: int = 20, events_path: str | Path | None = None) -> list[dict[str, Any]]:
    path = Path(events_path or DEFAULT_EVENTS_PATH)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in reversed(path.read_text(encoding='utf-8').splitlines()):
        payload = _safe_json_loads(line)
        if payload is None:
            continue
        rows.append(payload)
        if len(rows) >= max(1, int(limit)):
            break
    rows.reverse()
    return rows


def recent_session_digest(limit: int = 12, events_path: str | Path | None = None) -> list[dict[str, Any]]:
    rows = read_session_events(limit=limit * 3, events_path=events_path)
    by_session: dict[str, dict[str, Any]] = {}
    for row in rows:
        session_id = str(row.get('session_id') or '')
        if not session_id:
            continue
        current = by_session.setdefault(
            session_id,
            {
                'session_id': session_id,
                'interface': row.get('interface', ''),
                'last_ts': row.get('ts', ''),
                'events': [],
            },
        )
        current['last_ts'] = row.get('ts', current['last_ts'])
        current['events'].append(
            {
                'ts': row.get('ts', ''),
                'event_type': row.get('event_type', ''),
                'summary': row.get('summary', ''),
            }
        )
    ordered = sorted(by_session.values(), key=lambda item: str(item.get('last_ts', '')), reverse=True)
    for row in ordered:
        row['events'] = row['events'][-3:]
    return ordered[: max(1, int(limit))]


def _load_context_suggestions(query: str) -> dict[str, Any]:
    suggestions: dict[str, Any] = {}
    try:
        import sys
        for extra in (
            FORGE_ROOT / 'eidos_mcp' / 'src',
            FORGE_ROOT / 'memory_forge' / 'src',
            FORGE_ROOT / 'knowledge_forge' / 'src',
            FORGE_ROOT / 'lib',
        ):
            text = str(extra)
            if extra.exists() and text not in sys.path:
                sys.path.insert(0, text)
        from eidos_mcp.routers.tiered_memory import eidos_context_ingest, eidos_context_suggest
        from eidos_mcp.routers.knowledge import unified_context_search

        suggested = eidos_context_suggest(query, limit=3)
        unified = unified_context_search(query, max_results=4, include_memory=True, include_knowledge=True)
        try:
            suggestions['context_suggest'] = json.loads(suggested)
        except Exception:
            suggestions['context_suggest_raw'] = suggested
        try:
            suggestions['unified_context'] = json.loads(unified)
        except Exception:
            suggestions['unified_context_raw'] = unified
        suggestions['_ingest'] = eidos_context_ingest
    except Exception as exc:
        suggestions['error'] = str(exc)
    return suggestions


def build_session_context(
    *,
    interface: str,
    query: str,
    session_id: str,
    limit: int = 6,
    status_path: str | Path | None = None,
) -> dict[str, Any]:
    sync_external_sessions()
    digest = recent_session_digest(limit=limit)
    suggestions = _load_context_suggestions(query)
    payload = {
        'contract': 'eidos.session_context.v1',
        'generated_at': _now_iso(),
        'interface': str(interface),
        'session_id': str(session_id),
        'query': str(query),
        'recent_sessions': digest,
        'suggestions': {k: v for k, v in suggestions.items() if k != '_ingest'},
    }
    destination = Path(status_path or DEFAULT_STATUS_PATH)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + '\n', encoding='utf-8')
    return payload


def ingest_session_content(query: str, response: str | None = None) -> None:
    suggestions = _load_context_suggestions(query)
    ingest = suggestions.get('_ingest')
    if not callable(ingest):
        return
    try:
        ingest(query, 'prompt')
        if response:
            ingest(response, 'output')
    except Exception:
        return


def render_context_packet(payload: dict[str, Any], *, max_chars: int = 2400) -> str:
    lines: List[str] = []
    recent = payload.get('recent_sessions') or []
    if recent:
        lines.append('Recent Eidos sessions:')
        for session in recent[:4]:
            lines.append(f"- {session.get('interface')} {session.get('session_id')}: {session.get('events', [])[-1].get('summary', '') if session.get('events') else ''}".strip())
    suggestions = payload.get('suggestions') or {}
    context_suggest = suggestions.get('context_suggest') or {}
    if isinstance(context_suggest, dict):
        suggestion_map = context_suggest.get('suggestions') or {}
        if suggestion_map:
            lines.append('Relevant memory context:')
            for bucket, items in list(suggestion_map.items())[:3]:
                first = items[0]['content'] if items and isinstance(items[0], dict) else ''
                if first:
                    lines.append(f"- {bucket}: {first}")
    unified = suggestions.get('unified_context') or {}
    if isinstance(unified, dict):
        memory_hits = unified.get('memory_hits') or []
        knowledge_hits = unified.get('knowledge_hits') or []
        if memory_hits or knowledge_hits:
            lines.append(f"Unified context hits: memory={len(memory_hits)} knowledge={len(knowledge_hits)}")
    text = '\n'.join(line for line in lines if line).strip()
    if not text:
        text = 'No recent Eidos continuity context available.'
    return text[:max_chars]
