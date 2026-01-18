"""Generate a logbook entry using an LLM summary."""

from __future__ import annotations

import argparse
from pathlib import Path

from core.llm_adapter import LLMAdapter
from tools.logbook_entry import append_entry, LOGBOOK_PATH


def create_summary(text: str) -> str:
    """Return a single line summarizing ``text`` via :class:`LLMAdapter`."""
    adapter = LLMAdapter()
    return adapter.summarize(text)


def main(argv: list[str] | None = None) -> None:
    """Entry point for cycle summarization."""
    parser = argparse.ArgumentParser(
        description="Generate a cycle summary and append it to the logbook"
    )
    parser.add_argument("text", help="Cycle details to summarize")
    parser.add_argument("--next", dest="next_target", help="Optional next target")
    parser.add_argument(
        "--logbook", type=Path, default=LOGBOOK_PATH, help="Path to the logbook"
    )
    args = parser.parse_args(argv)

    summary = create_summary(args.text)
    append_entry(summary, args.next_target, args.logbook)


if __name__ == "__main__":
    main()
