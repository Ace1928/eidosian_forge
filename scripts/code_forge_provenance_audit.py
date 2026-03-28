from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _candidate_roots(repo_root: Path) -> list[Path]:
    roots = [
        repo_root / 'data' / 'code_forge',
        repo_root / 'archive_forge',
        repo_root / 'reports' / 'code_forge_eval',
    ]
    return [root for root in roots if root.exists()]


def _entry(path: Path, repo_root: Path, payload: dict[str, Any], *, kind: str) -> dict[str, Any]:
    links = payload.get('links') if isinstance(payload.get('links'), dict) else {}
    unit_links = links.get('unit_links') if isinstance(links, dict) and isinstance(links.get('unit_links'), list) else []
    return {
        'kind': kind,
        'path': str(path.relative_to(repo_root)),
        'generated_at': payload.get('generated_at'),
        'stage': payload.get('stage'),
        'root_path': payload.get('root_path'),
        'integration_run_id': payload.get('integration_run_id'),
        'source_run_id': payload.get('source_run_id'),
        'provenance_id': payload.get('provenance_id'),
        'registry_id': payload.get('registry_id'),
        'artifact_count': len(payload.get('artifacts') or []) if isinstance(payload.get('artifacts'), list) else 0,
        'unit_link_count': len(unit_links),
    }


def build_provenance_audit(repo_root: str | Path, limit: int = 12) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    link_entries: list[dict[str, Any]] = []
    registry_entries: list[dict[str, Any]] = []
    invalid_files: list[str] = []
    stage_counts: Counter[str] = Counter()

    for scan_root in _candidate_roots(root):
        for path in sorted(scan_root.rglob('provenance_*.json')):
            payload = _load_json(path)
            if not payload:
                invalid_files.append(str(path.relative_to(root)))
                continue
            if path.name == 'provenance_links.json':
                row = _entry(path, root, payload, kind='links')
                link_entries.append(row)
            elif path.name == 'provenance_registry.json':
                row = _entry(path, root, payload, kind='registry')
                registry_entries.append(row)
            else:
                continue
            stage_counts[str(payload.get('stage') or 'unknown')] += 1

    link_entries.sort(key=lambda row: str(row.get('generated_at') or ''), reverse=True)
    registry_entries.sort(key=lambda row: str(row.get('generated_at') or ''), reverse=True)
    latest_entries = sorted(
        link_entries + registry_entries,
        key=lambda row: str(row.get('generated_at') or ''),
        reverse=True,
    )[: max(1, int(limit))]

    return {
        'contract': 'eidos.code_forge_provenance_audit.v1',
        'generated_at': _now_iso(),
        'repo_root': str(root),
        'link_file_count': len(link_entries),
        'registry_file_count': len(registry_entries),
        'invalid_file_count': len(invalid_files),
        'stage_counts': dict(sorted(stage_counts.items())),
        'latest_generated_at': latest_entries[0]['generated_at'] if latest_entries else None,
        'latest_entries': latest_entries,
        'latest_links': link_entries[: max(1, int(limit))],
        'latest_registries': registry_entries[: max(1, int(limit))],
        'invalid_files': invalid_files[: max(1, int(limit))],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        '# Code Forge Provenance Audit',
        '',
        f"- Generated: `{report.get('generated_at')}`",
        f"- Link files: `{report.get('link_file_count')}`",
        f"- Registry files: `{report.get('registry_file_count')}`",
        f"- Invalid files: `{report.get('invalid_file_count')}`",
        f"- Latest generated_at: `{report.get('latest_generated_at')}`",
        '',
        '## Stage Counts',
        '',
    ]
    stage_counts = report.get('stage_counts') or {}
    if stage_counts:
        for stage, count in stage_counts.items():
            lines.append(f"- `{stage}`: `{count}`")
    else:
        lines.append('- None')
    lines.extend(['', '## Latest Entries', '', '| Kind | Stage | Generated | Unit Links | Path |', '| --- | --- | --- | ---: | --- |'])
    entries = report.get('latest_entries') or []
    if entries:
        for row in entries:
            lines.append(
                f"| {row.get('kind')} | {row.get('stage')} | {row.get('generated_at')} | {row.get('unit_link_count')} | {row.get('path')} |"
            )
    else:
        lines.append('| none | none | none | 0 | none |')
    return '\n'.join(lines) + '\n'


def write_provenance_audit(repo_root: str | Path, output_dir: str | Path, limit: int = 12) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    report = build_provenance_audit(root, limit=limit)
    json_path = report_dir / f'code_forge_provenance_audit_{stamp}.json'
    md_path = report_dir / f'code_forge_provenance_audit_{stamp}.md'
    latest_json = report_dir / 'latest.json'
    latest_md = report_dir / 'latest.md'
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    md_path.write_text(render_markdown(report), encoding='utf-8')
    latest_json.write_text(json_path.read_text(encoding='utf-8'), encoding='utf-8')
    latest_md.write_text(md_path.read_text(encoding='utf-8'), encoding='utf-8')
    return {
        'report': report,
        'json_path': str(json_path),
        'markdown_path': str(md_path),
        'latest_json': str(latest_json),
        'latest_markdown': str(latest_md),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description='Build a Code Forge provenance audit report.')
    parser.add_argument('--repo-root', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--limit', type=int, default=12)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (repo_root / 'reports' / 'code_forge_provenance_audit').resolve()
    result = write_provenance_audit(repo_root, output_dir, limit=max(1, int(args.limit)))
    print(json.dumps(result['report'], indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
