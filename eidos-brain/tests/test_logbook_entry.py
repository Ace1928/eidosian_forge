from datetime import datetime, timezone
from pathlib import Path
from tools.logbook_entry import append_entry


def test_append_entry(tmp_path: Path, monkeypatch) -> None:
    logbook = tmp_path / "log.md"
    logbook.write_text("# Eidos Logbook\n\n## Cycle 1: prior\n- old entry\n")

    fixed_time = datetime(2023, 1, 1, 12, 0, tzinfo=timezone.utc)

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_time

    monkeypatch.setattr("tools.logbook_entry.datetime", FixedDatetime)

    append_entry("new info", "next step", logbook)

    content = logbook.read_text()
    assert "## Cycle 2:" in content
    assert "- new info" in content
    assert "**Next Target:** next step" in content
