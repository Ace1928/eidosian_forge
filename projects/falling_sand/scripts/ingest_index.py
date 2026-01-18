"""CLI entrypoint wrapper for ingestion."""

from __future__ import annotations

from falling_sand.ingest import main


if __name__ == "__main__":
    raise SystemExit(main())
