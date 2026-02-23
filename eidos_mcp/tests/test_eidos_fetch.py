from __future__ import annotations

import json

import pytest

from eidos_mcp import eidos_fetch


async def _fake_fetch_stdio(_uri: str, _python: str):
    return {
        "success": True,
        "transport": "stdio",
        "uri": "eidos://persona",
        "content_count": 1,
        "contents": [{"kind": "text", "text": "persona payload"}],
    }


async def _fake_list_stdio(_python: str):
    return {
        "success": True,
        "transport": "stdio",
        "count": 2,
        "resources": [
            {"uri": "eidos://persona", "name": "Persona"},
            {"uri": "eidos://todo", "name": "TODO"},
        ],
    }


async def _fake_fetch_fail(_uri: str, _python: str):
    raise RuntimeError("boom")


def test_fetch_json_output(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(eidos_fetch, "_fetch_stdio", _fake_fetch_stdio)
    rc = eidos_fetch.main(["--transport", "stdio", "--json", "eidos://persona"])
    out = capsys.readouterr().out

    assert rc == 0
    payload = json.loads(out)
    assert payload["success"] is True
    assert payload["uri"] == "eidos://persona"


def test_list_plain_output(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(eidos_fetch, "_list_resources_stdio", _fake_list_stdio)
    rc = eidos_fetch.main(["--transport", "stdio", "--list"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "eidos://persona" in out
    assert "eidos://todo" in out


def test_error_json_output(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(eidos_fetch, "_fetch_stdio", _fake_fetch_fail)
    rc = eidos_fetch.main(["--transport", "stdio", "--json-errors", "eidos://persona"])
    out = capsys.readouterr().out

    assert rc == 1
    payload = json.loads(out)
    assert payload["success"] is False
    assert "boom" in payload["error"]
