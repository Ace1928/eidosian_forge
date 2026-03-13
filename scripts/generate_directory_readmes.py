#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FORGE_ROOT = Path(__file__).resolve().parent.parent
for extra in (
    FORGE_ROOT / "doc_forge" / "src",
    FORGE_ROOT / "lib",
    FORGE_ROOT,
):
    value = str(extra)
    if extra.exists() and value not in sys.path:
        sys.path.insert(0, value)

from doc_forge.scribe.directory_docs import (  # type: ignore
    inventory_summary,
    record_map,
    upsert_directory_readme,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate managed README.md files for documented directories.")
    parser.add_argument("--repo-root", default=str(FORGE_ROOT))
    parser.add_argument("--path", action="append", default=[], help="specific relative directory to document")
    parser.add_argument("--missing-only", action="store_true", help="only write directories that lack a README")
    parser.add_argument(
        "--managed-only",
        action="store_true",
        help="only rewrite README files already managed by the Eidos documentation contract",
    )
    parser.add_argument("--limit", type=int, default=0, help="optional maximum number of directories to write")
    parser.add_argument(
        "--summary-json",
        default=str(FORGE_ROOT / "reports" / "docs" / "directory_docs_summary.json"),
        help="where to write the inventory summary artifact",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    selected = set(args.path or [])
    summary = inventory_summary(repo_root, selected_paths=selected or None)
    records = summary["records"]
    records_by_path = record_map(repo_root, selected_paths=selected or None)
    writes = []
    for record in records:
        rel_dir = record["path"]
        if selected and not any(
            rel_dir == prefix or rel_dir.startswith(prefix + "/") or prefix.startswith(rel_dir + "/")
            for prefix in selected
        ):
            continue
        readme_path = repo_root / rel_dir / "README.md"
        is_managed = readme_path.exists() and "EIDOS:DOCSYS:START" in readme_path.read_text(
            encoding="utf-8", errors="ignore"
        )
        if args.missing_only and record["has_readme"]:
            continue
        if args.managed_only and not is_managed:
            continue
        writes.append(rel_dir)
    if args.limit and args.limit > 0:
        writes = writes[: args.limit]

    results = []
    for rel_dir in writes:
        results.append(upsert_directory_readme(repo_root, rel_dir, records=records_by_path))

    summary_path = Path(args.summary_json).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary["write_count"] = len(results)
    summary["writes"] = [
        {
            "path": item["path"],
            "readme_path": item["readme_path"],
            "changed": item["changed"],
            "created": item["created"],
        }
        for item in results
    ]
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), "write_count": len(results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
