# Current State: moltbook_forge

**Date**: 2026-02-05
**Status**: Operational

## Summary
- Safety-first pipeline for Moltbook content (normalize -> validate -> screen -> quarantine).
- CLI wrapper available via `python -m moltbook_forge` and `eidosian moltbook`.
- Network ingestion is allowlist-only and requires explicit `--allow-network`.

## Key Scripts
- `moltbook_sanitize.py`: normalize and flag untrusted content.
- `moltbook_screen.py`: policy gate with critical flags and risk threshold.
- `moltbook_validate.py`: schema validation for sanitized payloads.
- `moltbook_quarantine.py`: quarantines high-risk payloads.
- `moltbook_bootstrap.py`: end-to-end snapshot of skill content.
- `moltbook_skill_review.py`: skill review with redaction and extraction.

## Tests
- `moltbook_forge/tests`: 25 passing tests (CLI + safety pipeline).
