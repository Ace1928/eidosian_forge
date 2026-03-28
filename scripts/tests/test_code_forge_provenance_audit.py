from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / 'code_forge_provenance_audit.py'


def _load_module():
    spec = importlib.util.spec_from_file_location('code_forge_provenance_audit', SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding='utf-8')


def test_build_provenance_audit_counts_links_and_registries(tmp_path: Path) -> None:
    module = _load_module()
    repo = tmp_path
    _write_json(
        repo / 'data' / 'code_forge' / 'ingestion_runs' / 'run1' / 'provenance_links.json',
        {
            'generated_at': '2026-03-20T00:00:00Z',
            'stage': 'digester',
            'root_path': '/repo',
            'provenance_id': 'prov-1',
            'links': {'unit_links': [{'unit_id': 'u1'}]},
        },
    )
    _write_json(
        repo / 'data' / 'code_forge' / 'ingestion_runs' / 'run1' / 'provenance_registry.json',
        {
            'generated_at': '2026-03-20T00:01:00Z',
            'stage': 'digester',
            'root_path': '/repo',
            'registry_id': 'reg-1',
            'links': {'unit_links': [{'unit_id': 'u1'}, {'unit_id': 'u2'}]},
        },
    )

    report = module.build_provenance_audit(repo)

    assert report['contract'] == 'eidos.code_forge_provenance_audit.v1'
    assert report['link_file_count'] == 1
    assert report['registry_file_count'] == 1
    assert report['stage_counts']['digester'] == 2
    assert report['latest_entries'][0]['kind'] == 'registry'
    assert report['latest_entries'][0]['unit_link_count'] == 2


def test_write_provenance_audit_writes_latest_aliases(tmp_path: Path) -> None:
    module = _load_module()
    repo = tmp_path / 'repo'
    _write_json(
        repo / 'data' / 'code_forge' / 'roundtrip' / 'r1' / 'provenance_links.json',
        {'generated_at': '2026-03-20T00:00:00Z', 'stage': 'roundtrip', 'root_path': '/repo'},
    )

    result = module.write_provenance_audit(repo, repo / 'reports' / 'code_forge_provenance_audit')

    assert Path(result['json_path']).exists()
    assert Path(result['markdown_path']).exists()
    assert Path(result['latest_json']).exists()
    assert Path(result['latest_markdown']).exists()
    latest = json.loads(Path(result['latest_json']).read_text(encoding='utf-8'))
    assert latest['stage_counts']['roundtrip'] == 1
