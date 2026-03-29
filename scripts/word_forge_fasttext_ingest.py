from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _setup_imports(repo_root: Path) -> None:
    candidates = [repo_root / "word_forge" / "src", repo_root / "lib"]
    for candidate in candidates:
        candidate_text = str(candidate)
        if candidate.exists() and candidate_text not in sys.path:
            sys.path.insert(0, candidate_text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Word Forge FastText aligned-vector ingestion.")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--source-path", required=True)
    parser.add_argument("--lang", required=True)
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--vector-db-path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--bootstrap-lang", default=None)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--min-score", type=float, default=0.55)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    _setup_imports(repo_root)

    from word_forge.multilingual.fasttext_runtime import run_fasttext_ingest

    db_path = Path(args.db_path).resolve() if args.db_path else (repo_root / "word_forge" / "data" / "word_forge.sqlite")
    vector_db_path = (
        Path(args.vector_db_path).resolve() if args.vector_db_path else (repo_root / "data" / "word_forge_fasttext.sqlite")
    )
    result = run_fasttext_ingest(
        repo_root=repo_root,
        source_path=Path(args.source_path).resolve(),
        lang=args.lang,
        db_path=db_path,
        vector_db_path=vector_db_path,
        limit=args.limit,
        bootstrap_lang=args.bootstrap_lang,
        top_k=args.top_k,
        min_score=args.min_score,
        apply=args.apply,
        force=args.force,
    )
    print(json.dumps(result["status"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
