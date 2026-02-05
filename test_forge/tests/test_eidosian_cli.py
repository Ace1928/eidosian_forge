from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_eidosian_forges_json() -> None:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "bin/eidosian", "--json", "forges"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    names = {forge["name"]: forge for forge in payload["forges"]}
    assert "moltbook" in names
    assert names["moltbook"]["available"] is True
