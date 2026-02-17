from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[2]
    script_path = root / "scripts" / "generate_directory_atlas.py"
    spec = importlib.util.spec_from_file_location("generate_directory_atlas", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_atlas_generation_contains_expected_sections() -> None:
    mod = _load_module()
    root = Path(__file__).resolve().parents[2]
    all_dirs = mod._collect_all_dirs(root)
    tracked = mod._run_git_ls_files(root)

    atlas = mod.build_atlas(root, all_dirs, tracked, max_depth=2)

    assert "# Eidosian Forge Directory Atlas" in atlas
    assert "## Top-Level Overview" in atlas
    assert "## Directory Inventory (Depth-Limited)" in atlas
    assert "docs/DIRECTORY_INDEX_FULL.txt" in atlas
    assert "`agent_forge`" in atlas


def test_full_index_writer_is_deterministic_shape(tmp_path: Path) -> None:
    mod = _load_module()
    out = tmp_path / "DIRECTORY_INDEX_FULL.txt"
    dirs = ["alpha", "alpha/beta", "gamma"]

    mod.write_full_index(out, dirs)

    text = out.read_text(encoding="utf-8")
    assert "# Full Recursive Directory Index" in text
    assert "Directory count: 3" in text
    assert "alpha" in text
    assert "alpha/beta" in text
    assert "gamma" in text
