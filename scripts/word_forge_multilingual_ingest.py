from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _setup_imports(repo_root: Path) -> None:
    candidates = [
        repo_root / "word_forge" / "src",
        repo_root / "lib",
    ]
    for candidate in candidates:
        candidate_text = str(candidate)
        if candidate.exists() and candidate_text not in sys.path:
            sys.path.insert(0, candidate_text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run incremental Word Forge multilingual ingestion.")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--source-path", required=True)
    parser.add_argument("--source-type", choices=("kaikki", "wiktextract"), required=True)
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    _setup_imports(repo_root)

    from word_forge.multilingual.runtime import run_multilingual_ingest

    source_path = Path(args.source_path).resolve()
    db_path = Path(args.db_path).resolve() if args.db_path else (repo_root / "word_forge" / "data" / "word_forge.sqlite")
    result = run_multilingual_ingest(
        repo_root=repo_root,
        source_path=source_path,
        source_type=args.source_type,
        db_path=db_path,
        limit=args.limit,
        force=args.force,
    )
    print(json.dumps(result["status"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
