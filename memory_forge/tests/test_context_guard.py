from __future__ import annotations

import time
from pathlib import Path

from memory_forge.context.guard import redact_sensitive, retention_cleanup


def test_redact_sensitive() -> None:
    text = "api_key=secret moltbook_sk_abc12345 192.168.0.1 test@example.com"
    redacted, findings = redact_sensitive(text)
    assert "[REDACTED:MOLTBOOK_API_KEY]" in redacted
    assert "[REDACTED:IP_ADDRESS]" in redacted
    assert any(f.label == "email_address" for f in findings)


def test_retention_cleanup(tmp_path: Path) -> None:
    old_file = tmp_path / "old.txt"
    new_file = tmp_path / "new.txt"
    old_file.write_text("old", encoding="utf-8")
    new_file.write_text("new", encoding="utf-8")
    past = time.time() - (10 * 86400)
    import os

    os.utime(old_file, (past, past))
    removed = retention_cleanup(tmp_path, retention_days=5, dry_run=False)
    assert str(old_file) in removed
    assert new_file.exists()
