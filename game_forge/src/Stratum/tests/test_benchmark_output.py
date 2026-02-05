from __future__ import annotations

import json
from pathlib import Path

from tests import benchmark as bench


def test_format_single_payload() -> None:
    result = {"grid_size": 8, "ticks_per_second": 1.0}
    payload = bench.format_single_payload(result, ticks=5)
    assert payload["mode"] == "single"
    assert payload["grid_size"] == 8
    assert payload["ticks"] == 5
    assert payload["result"]["ticks_per_second"] == 1.0


def test_format_scaling_payload() -> None:
    results = [{"grid_size": 8}, {"grid_size": 16}]
    payload = bench.format_scaling_payload(results, ticks=3)
    assert payload["mode"] == "scaling"
    assert payload["ticks"] == 3
    assert len(payload["results"]) == 2


def test_write_payload(tmp_path: Path) -> None:
    output = tmp_path / "bench.json"
    bench.write_payload(str(output), {"mode": "single"})
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["mode"] == "single"
