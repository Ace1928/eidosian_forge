from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "download_local_models.py"
SPEC = importlib.util.spec_from_file_location("download_local_models", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)  # type: ignore[assignment]


def test_resolve_specs_filters_by_profile() -> None:
    catalog = {
        "models": [
            {"id": "a", "profiles": ["core"]},
            {"id": "b", "profiles": ["toolcalling"]},
            {"id": "c", "profiles": ["core", "toolcalling"]},
        ]
    }
    picked = mod._resolve_specs(catalog, "toolcalling", set())
    assert [m["id"] for m in picked] == ["b", "c"]


def test_resolve_specs_filters_by_model_id_override() -> None:
    catalog = {"models": [{"id": "a", "profiles": ["core"]}, {"id": "b", "profiles": ["core"]}]}
    picked = mod._resolve_specs(catalog, "core", {"b"})
    assert [m["id"] for m in picked] == ["b"]


def test_artifact_specs_includes_aux_files() -> None:
    item = {
        "id": "vision",
        "repo": "repo/main",
        "filename": "model.gguf",
        "path": "models/model.gguf",
        "aux_files": [{"repo": "repo/main", "filename": "mmproj.gguf", "path": "models/mmproj.gguf"}],
    }
    specs = mod._artifact_specs(item, Path("models"))
    assert len(specs) == 2
    assert specs[0].filename == "model.gguf"
    assert specs[1].filename == "mmproj.gguf"


def test_read_catalog_validates_shape(tmp_path: Path) -> None:
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({"models": [{"id": "x"}]}))
    payload = mod._read_catalog(path)
    assert isinstance(payload["models"], list)
