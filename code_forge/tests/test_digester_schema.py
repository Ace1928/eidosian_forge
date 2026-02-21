import json
from pathlib import Path

from code_forge.digester.pipeline import run_archive_digester
from code_forge.digester.schema import validate_output_dir
from code_forge.ingest.runner import IngestionRunner
from code_forge.library.db import CodeLibraryDB


def _make_repo(root: Path) -> None:
    (root / "src").mkdir(parents=True)
    (root / "src" / "m.py").write_text(
        "import math\n" "def helper(v):\n" "    return math.floor(v)\n",
        encoding="utf-8",
    )


def test_validate_output_dir_pass_and_fail(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _make_repo(repo)

    db = CodeLibraryDB(tmp_path / "library.sqlite")
    runner = IngestionRunner(db=db, runs_dir=tmp_path / "runs")
    out = tmp_path / "digester"

    run_archive_digester(
        root_path=repo,
        db=db,
        runner=runner,
        output_dir=out,
        mode="analysis",
        extensions=[".py"],
        progress_every=1,
    )

    ok = validate_output_dir(out)
    assert ok["pass"]

    triage_path = out / "triage.json"
    payload = json.loads(triage_path.read_text(encoding="utf-8"))
    payload.pop("entries", None)
    triage_path.write_text(json.dumps(payload), encoding="utf-8")

    bad = validate_output_dir(out)
    assert not bad["pass"]
    assert any("triage.json" in err for err in bad["errors"])

    # Restore triage and break provenance registry to ensure optional schema validation is enforced.
    run_archive_digester(
        root_path=repo,
        db=db,
        runner=runner,
        output_dir=out,
        mode="analysis",
        extensions=[".py"],
        progress_every=1,
    )
    registry_path = out / "provenance_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["links"] = []  # invalid shape; must be object
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    bad_registry = validate_output_dir(out)
    assert bad_registry["pass"] is False
    assert any("provenance_registry.json" in err for err in bad_registry["errors"])
