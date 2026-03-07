#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FORGE_ROOT = Path(__file__).resolve().parent.parent
for extra in (FORGE_ROOT / "lib", FORGE_ROOT / "eidos_mcp" / "src"):
    value = str(extra)
    if extra.exists() and value not in sys.path:
        sys.path.insert(0, value)

from eidosian_agent.local_mcp_agent import DEFAULT_POLICY_PATH, cli_entry


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Guarded local MCP-backed agent for Qwen + Eidos MCP.")
    parser.add_argument("objective", help="Objective for the local agent cycle.")
    parser.add_argument("--profile", default="observer", help="Policy profile name from cfg/local_agent_profiles.json")
    parser.add_argument("--policy-path", default=str(DEFAULT_POLICY_PATH))
    parser.add_argument("--model", default="qwen3.5:2b")
    parser.add_argument("--model-url", default="")
    parser.add_argument("--mcp-url", default="")
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--interval-sec", type=float, default=120.0)
    parser.add_argument("--max-cycles", type=int, default=0)
    parser.add_argument("--timeout-sec", type=float, default=1800.0)
    parser.add_argument("--keep-alive", default="", help="Override Ollama keep_alive residency, e.g. 2h.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = cli_entry(
        objective=str(args.objective),
        profile_name=str(args.profile),
        policy_path=str(args.policy_path),
        model=str(args.model),
        model_url=str(args.model_url or ""),
        mcp_url=str(args.mcp_url or ""),
        continuous=bool(args.continuous),
        interval_sec=float(args.interval_sec),
        max_cycles=int(args.max_cycles),
        timeout_sec=float(args.timeout_sec),
        keep_alive=str(args.keep_alive or ""),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if str(result.get("status") or "") in {"success", "blocked", "timeout"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
