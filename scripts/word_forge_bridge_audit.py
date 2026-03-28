from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _setup_imports(repo_root: Path) -> None:
    candidates = [
        repo_root / "word_forge" / "src",
        repo_root / "lib",
        repo_root / "scripts",
    ]
    for candidate in candidates:
        candidate_text = str(candidate)
        if candidate.exists() and candidate_text not in sys.path:
            sys.path.insert(0, candidate_text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a Word Forge bridge audit report.")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--db-path", default=None)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    _setup_imports(repo_root)

    from word_forge.bridge.audit import run_bridge_audit

    db_path = Path(args.db_path).resolve() if args.db_path else None
    result = run_bridge_audit(repo_root=repo_root, db_path=db_path)
    print(json.dumps(result["status"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
