from pathlib import Path

import tools.cycle_summary as cycle_summary


def test_main_writes_log(monkeypatch, tmp_path: Path) -> None:
    called = {}

    def fake_create_summary(text: str) -> str:
        called["text"] = text
        return "summary line"

    def fake_append(msg: str, nxt: str | None, path: Path) -> None:
        called["msg"] = msg
        called["next"] = nxt
        called["path"] = path

    monkeypatch.setattr(cycle_summary, "create_summary", fake_create_summary)
    monkeypatch.setattr(cycle_summary, "append_entry", fake_append)

    logbook = tmp_path / "log.md"
    cycle_summary.main(
        ["details for cycle", "--next", "future", "--logbook", str(logbook)]
    )

    assert called["text"] == "details for cycle"
    assert called["msg"] == "summary line"
    assert called["next"] == "future"
    assert called["path"] == logbook
