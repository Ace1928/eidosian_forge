"""Append timestamped cycles to the Eidos logbook."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

LOGBOOK_PATH = Path("knowledge/eidos_logbook.md")


def read_logbook(path: Path = LOGBOOK_PATH) -> str:
    """Return the logbook contents or an empty template."""
    if path.exists():
        return path.read_text()
    return "# Eidos Logbook\n"


def next_cycle_number(text: str) -> int:
    """Return the next cycle number found in ``text``."""
    last = 0
    for line in text.splitlines():
        if line.startswith("## Cycle"):
            try:
                num = int(line.split()[2].rstrip(":"))
                last = max(last, num)
            except ValueError:
                continue
    return last + 1


def format_entry(cycle: int, message: str, next_target: str | None = None) -> str:
    """Return a formatted logbook entry."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"## Cycle {cycle}: {timestamp}", f"- {message}"]
    if next_target:
        lines.append("")
        lines.append(f"**Next Target:** {next_target}")
    lines.append("")
    return "\n".join(lines)


def append_entry(
    message: str, next_target: str | None = None, path: Path = LOGBOOK_PATH
) -> None:
    """Append an entry to the logbook at ``path``."""
    text = read_logbook(path)
    cycle = next_cycle_number(text)
    entry = format_entry(cycle, message, next_target)
    updated = text.rstrip() + "\n\n" + entry
    path.write_text(updated)


def main() -> None:
    """Entry point for adding logbook entries."""
    parser = argparse.ArgumentParser(description="Append an entry to the logbook")
    parser.add_argument("message", help="Summary line for the logbook entry")
    parser.add_argument("--next", dest="next_target", help="Optional next target")
    parser.add_argument(
        "--logbook",
        type=Path,
        default=LOGBOOK_PATH,
        help="Path to the logbook file",
    )
    args = parser.parse_args()
    append_entry(args.message, args.next_target, args.logbook)


if __name__ == "__main__":
    main()
