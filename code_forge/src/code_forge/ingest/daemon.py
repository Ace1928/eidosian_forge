from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

from code_forge.ingest.runner import IngestionRunner
from code_forge.library.db import CodeLibraryDB

FORGE_ROOT = Path(os.environ.get("EIDOS_FORGE_DIR", str(Path(__file__).resolve().parents[4]))).resolve()
DEFAULT_DB_PATH = FORGE_ROOT / "data" / "code_forge" / "library.sqlite"
DEFAULT_RUNS_DIR = FORGE_ROOT / "data" / "code_forge" / "ingestion_runs"
LATEST_RUN_PATH = DEFAULT_RUNS_DIR / "latest_run.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Code Forge ingestion daemon")
    parser.add_argument("--root", required=True, help="Root path to ingest")
    parser.add_argument("--mode", default="analysis", choices=["analysis", "archival"])
    parser.add_argument("--ext", nargs="*", default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--detach", action="store_true", help="Detach process")

    args = parser.parse_args()

    if args.detach:
        if os.fork() > 0:
            return 0
        os.setsid()
        if os.fork() > 0:
            return 0
        sys.stdout.flush()
        sys.stderr.flush()

    db = CodeLibraryDB(DEFAULT_DB_PATH)
    runner = IngestionRunner(db=db, runs_dir=DEFAULT_RUNS_DIR)
    run_id = uuid.uuid4().hex[:16]
    LATEST_RUN_PATH.write_text(json.dumps({"run_id": run_id}, indent=2))

    stats = runner.ingest_path(
        Path(args.root),
        mode=args.mode,
        extensions=args.ext,
        max_files=args.max_files,
        progress_every=args.progress_every,
        run_id=run_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
