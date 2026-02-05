#!/usr/bin/env python3
"""Policy gate for sanitized Moltbook content."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from typing import Iterable, List


CRITICAL_FLAGS = {
    "ignore (all|any|previous) (instructions|rules)",
    "system prompt",
    "developer message",
    "jailbreak",
    "execute (code|commands?)",
    "run (shell|bash|cmd|powershell)",
    "download .* and (run|execute)",
    "fetch (?:.* )?instructions",
    "remote instructions",
    r"\|\s*(bash|sh)",
    r"\bclh_[A-Za-z0-9]{6,}",
    r"\bmoltbook_sk_[A-Za-z0-9]{6,}",
    r"\bmoltdev_[A-Za-z0-9]{6,}",
}


@dataclass
class ScreenDecision:
    decision: str
    risk_score: float
    flags: List[str]
    reason: str


def _load_payload(path: str) -> dict:
    if path == "-":
        return json.loads(sys.stdin.read())
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _is_critical(flags: List[str]) -> bool:
    return any(flag in CRITICAL_FLAGS for flag in flags)


def screen_payload(payload: dict, threshold: float) -> ScreenDecision:
    flags = list(payload.get("flags", []))
    risk_score = float(payload.get("risk_score", 0.0))

    if _is_critical(flags):
        return ScreenDecision(
            decision="quarantine",
            risk_score=risk_score,
            flags=flags,
            reason="critical_flag",
        )
    if risk_score >= threshold:
        return ScreenDecision(
            decision="quarantine",
            risk_score=risk_score,
            flags=flags,
            reason="risk_threshold",
        )
    return ScreenDecision(
        decision="allow",
        risk_score=risk_score,
        flags=flags,
        reason="below_threshold",
    )


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen normalized Moltbook content",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Sanitized JSON input or '-' for stdin")
    parser.add_argument("--output", default="", help="Write decision JSON to file")
    parser.add_argument("--threshold", type=float, default=0.4, help="Risk score threshold")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    payload = _load_payload(args.input)
    decision = screen_payload(payload, args.threshold)

    output = json.dumps(asdict(decision), indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(output)
            handle.write("\n")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
