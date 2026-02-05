"""Ingest index documents into SQLite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from falling_sand import schema
from falling_sand.db import DbConfig, connect_db, ingest_document, migrate_db
from falling_sand.models import index_document_from_dict
from eidosian_core import eidosian


@eidosian()
def build_parser() -> argparse.ArgumentParser:
    """Create CLI arguments for ingestion."""

    parser = argparse.ArgumentParser(description="Ingest index.json into SQLite.")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=Path("artifacts/index.db"))
    parser.add_argument("--batch-size", type=int, default=1000)
    return parser


@eidosian()
def ingest_index(index_path: Path, db_path: Path, batch_size: int = 1000) -> int:
    """Ingest index document from disk into SQLite and return the run ID."""

    if not index_path.exists():
        raise ValueError(f"Index file not found: {index_path}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    payload = schema.migrate_document_dict(payload)
    document = index_document_from_dict(payload)

    config = DbConfig(path=db_path, batch_size=batch_size)
    connection = connect_db(config)
    try:
        migrate_db(connection)
        run_id = ingest_document(connection, document, batch_size=config.batch_size)
    finally:
        connection.close()

    return run_id


@eidosian()
def main(argv: Sequence[str] | None = None) -> int:
    """Run the ingestion CLI."""

    args = build_parser().parse_args(argv)
    ingest_index(args.index, args.db, batch_size=args.batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
