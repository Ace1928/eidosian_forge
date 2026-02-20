from __future__ import annotations

import json
from pathlib import Path

from code_forge.ingest.runner import IngestionRunner
from code_forge.integration.pipeline import export_units_for_graphrag, sync_units_to_knowledge_forge
from code_forge.library.db import CodeLibraryDB


def _make_repo(root: Path) -> None:
    (root / "src").mkdir(parents=True)
    (root / "src" / "m.py").write_text(
        "def add(a, b):\n" "    return a + b\n",
        encoding="utf-8",
    )


def test_export_units_for_graphrag_writes_manifest(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _make_repo(repo)

    db = CodeLibraryDB(tmp_path / "library.sqlite")
    runner = IngestionRunner(db=db, runs_dir=tmp_path / "runs")
    stats = runner.ingest_path(repo, extensions=[".py"], progress_every=1)

    out = tmp_path / "grag"
    payload = export_units_for_graphrag(
        db=db,
        output_dir=out,
        limit=200,
        min_token_count=1,
        run_id=str(stats.run_id),
    )

    manifest_path = Path(payload["manifest_path"])
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_id"] == str(stats.run_id)
    assert manifest["generated_documents"] == payload["exported"]
    assert isinstance(manifest.get("documents"), list)


def test_sync_units_to_knowledge_forge_node_links(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _make_repo(repo)

    db = CodeLibraryDB(tmp_path / "library.sqlite")
    runner = IngestionRunner(db=db, runs_dir=tmp_path / "runs")
    stats = runner.ingest_path(repo, extensions=[".py"], progress_every=1)

    kb_path = tmp_path / "kb.json"
    payload = sync_units_to_knowledge_forge(
        db=db,
        kb_path=kb_path,
        limit=200,
        min_token_count=1,
        run_id=str(stats.run_id),
        include_node_links=True,
        node_links_limit=50,
    )

    assert payload["run_id"] == str(stats.run_id)
    assert isinstance(payload["node_links"], list)
    assert payload["node_links"]
