import json
from pathlib import Path

from code_forge.canonicalize.planner import build_canonical_migration_plan


def test_build_canonical_migration_plan(tmp_path: Path) -> None:
    triage = {
        "entries": [
            {
                "file_path": "src/core/alpha.py",
                "label": "extract",
                "reasons": ["duplicate pressure"],
                "metrics": {"language": "python", "category": "source"},
            },
            {
                "file_path": "tests/test_alpha.py",
                "label": "refactor",
                "reasons": ["complexity"],
                "metrics": {"language": "python", "category": "test"},
            },
            {
                "file_path": "docs/notes.md",
                "label": "keep",
                "reasons": ["documentation"],
                "metrics": {"language": "markdown", "category": "doc"},
            },
        ]
    }
    triage_path = tmp_path / "triage.json"
    triage_path.write_text(json.dumps(triage), encoding="utf-8")

    out = tmp_path / "canonical"
    summary = build_canonical_migration_plan(
        triage_path=triage_path,
        output_dir=out,
        include_labels=["extract", "refactor"],
        max_entries=10,
        generate_shims=True,
    )

    assert summary["selected_count"] == 2
    assert (out / "migration_map.json").exists()
    assert (out / "canonicalization_plan.md").exists()
    assert (out / "canonicalization_summary.json").exists()

    migration = json.loads((out / "migration_map.json").read_text(encoding="utf-8"))
    assert migration["entries"]
    targets = {entry["target_path"] for entry in migration["entries"]}
    assert any(path.startswith("canonical/src/python/") for path in targets)
    assert any(path.startswith("canonical/tests/python/") for path in targets)
