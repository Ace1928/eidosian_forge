from __future__ import annotations

import json

from eidos_mcp.check_mcp import main


def test_check_mcp_json_includes_component_checks(capsys) -> None:
    exit_code = main(["--json"])
    captured = capsys.readouterr()
    assert exit_code == 0

    payload = json.loads(captured.out)
    assert payload["tool_count"] >= 1
    assert payload["resource_count"] >= 1
    assert "component_checks" in payload
    assert "mcp_version" in payload["component_checks"]


def test_check_mcp_smoke_test_in_json(capsys) -> None:
    exit_code = main(["--json", "--smoke-test"])
    captured = capsys.readouterr()
    assert exit_code == 0

    payload = json.loads(captured.out)
    smoke = payload["component_checks"]["server_smoke_build"]
    assert smoke["ok"] is True
