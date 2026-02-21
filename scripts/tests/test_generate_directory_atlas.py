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


def test_atlas_generation_contains_expected_sections_tracked_scope() -> None:
    mod = _load_module()
    root = Path(__file__).resolve().parents[2]
    tracked = mod._run_git_ls_files(root)
    tracked_dirs = mod._tracked_dirs_from_files(tracked)

    atlas = mod.build_atlas(
        root,
        tracked_dirs,
        tracked,
        max_depth=2,
        scope="tracked",
        generated_at=None,
    )

    assert "# Eidosian Forge Directory Atlas" in atlas
    assert "## Top-Level Overview" in atlas
    assert "## Directory Inventory (Depth-Limited)" in atlas
    assert "docs/DIRECTORY_INDEX_FULL.txt" in atlas
    assert "- Scope: `tracked`" in atlas
    assert "deterministic (timestamp omitted)" in atlas
    assert "`agent_forge`" in atlas


def test_filesystem_collection_hidden_controls(tmp_path: Path) -> None:
    mod = _load_module()

    (tmp_path / "alpha").mkdir()
    (tmp_path / "alpha" / "beta").mkdir()
    (tmp_path / ".github").mkdir()
    (tmp_path / ".local_state").mkdir()
    (tmp_path / ".git").mkdir()

    no_hidden = mod._collect_filesystem_dirs(tmp_path, include_hidden_top=False)
    assert "alpha" in no_hidden
    assert "alpha/beta" in no_hidden
    assert ".github" in no_hidden
    assert ".local_state" not in no_hidden
    assert ".git" not in no_hidden

    with_hidden = mod._collect_filesystem_dirs(tmp_path, include_hidden_top=True)
    assert ".local_state" in with_hidden
    assert ".git" not in with_hidden


def test_runtime_top_level_dirs_hidden_controls(tmp_path: Path) -> None:
    mod = _load_module()
    (tmp_path / "docs").mkdir()
    (tmp_path / ".vscode").mkdir()
    (tmp_path / ".cachetmp").mkdir()
    (tmp_path / ".git").mkdir()

    no_hidden = mod._runtime_top_level_dirs(tmp_path, include_hidden_top=False)
    assert "docs" in no_hidden
    assert ".vscode" in no_hidden
    assert ".cachetmp" not in no_hidden
    assert ".git" not in no_hidden

    with_hidden = mod._runtime_top_level_dirs(tmp_path, include_hidden_top=True)
    assert ".cachetmp" in with_hidden
    assert ".git" not in with_hidden


def test_generated_at_resolution() -> None:
    mod = _load_module()
    assert mod._resolve_generated_at("") is None
    assert mod._resolve_generated_at("  ") is None
    assert mod._resolve_generated_at("2026-02-17T00:00:00Z") == "2026-02-17T00:00:00Z"
    now_value = mod._resolve_generated_at("now")
    assert isinstance(now_value, str)
    assert now_value.endswith("Z")


def test_readme_detection_respects_tracked_scope(tmp_path: Path) -> None:
    mod = _load_module()
    (tmp_path / "eidosian_venv").mkdir()
    (tmp_path / "eidosian_venv" / "README.md").write_text("local readme", encoding="utf-8")

    # With tracked scope, local-only README should not be emitted.
    assert mod._readme_for_dir(tmp_path, "eidosian_venv", tracked_files=set()) == ""

    # Filesystem scope still reports README when present.
    assert mod._readme_for_dir(tmp_path, "eidosian_venv", tracked_files=None) == "eidosian_venv/README.md"


def test_full_index_writer_deterministic_shape(tmp_path: Path) -> None:
    mod = _load_module()
    out = tmp_path / "DIRECTORY_INDEX_FULL.txt"
    dirs = ["alpha", "alpha/beta", "gamma"]

    mod.write_full_index(out, dirs, scope="tracked", generated_at=None)

    text = out.read_text(encoding="utf-8")
    assert "# Full Recursive Directory Index" in text
    assert "Generated: deterministic (timestamp omitted)" in text
    assert "Scope: tracked" in text
    assert "Directory count: 3" in text
    assert "alpha" in text
    assert "alpha/beta" in text
    assert "gamma" in text
