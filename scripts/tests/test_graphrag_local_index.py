from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "graphrag_local_index.py"


def _load_module():
    loader = importlib.machinery.SourceFileLoader(
        "graphrag_local_index",
        str(SCRIPT_PATH),
    )
    spec = importlib.util.spec_from_loader("graphrag_local_index", loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    loader.exec_module(module)
    return module


mod = _load_module()


def test_has_existing_output_accepts_native_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "native_community_reports.json").write_text(json.dumps({"reports": [{}]}), encoding="utf-8")

    assert mod.has_existing_output(tmp_path) is True


def test_validate_index_output_accepts_native_report_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "stats.json").write_text(json.dumps({"num_documents": 2, "update_documents": 0}), encoding="utf-8")
    (output_dir / "native_community_reports.json").write_text(
        json.dumps({"reports": [{"title": "Docs", "summary": "ok"}]}),
        encoding="utf-8",
    )

    mod.validate_index_output(tmp_path)
