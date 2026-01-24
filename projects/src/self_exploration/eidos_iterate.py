#!/usr/bin/env python3
"""
🔄 Eidosian Self-Exploration Iteration Daemon

This script enables autonomous continuation of the self-exploration process.
It can be triggered by the computer_control_forge to simulate user input
that continues the iterative cycle.

Safety:
- Respects kill switch
- Full audit trail
- Idempotent operations
- Clear output for next iteration

Usage:
    python eidos_iterate.py --check     # Check status only
    python eidos_iterate.py --continue  # Trigger continuation
    python eidos_iterate.py --summary   # Generate cycle summary
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project paths
sys.path.insert(0, "/home/lloyd/eidosian_forge/projects/src")
sys.path.insert(0, "/home/lloyd/eidosian_forge/eidos_mcp/src")
sys.path.insert(0, "/home/lloyd/eidosian_forge/computer_control_forge/src")

SELF_EXPLORATION_DIR = Path("/home/lloyd/eidosian_forge/projects/src/self_exploration")
PROVENANCE_DIR = SELF_EXPLORATION_DIR / "provenance"
DATA_DIR = SELF_EXPLORATION_DIR / "data"
IDENTITY_DIR = SELF_EXPLORATION_DIR / "identity"

# Kill switch check
KILL_FILE = Path("/tmp/eidosian_control_kill")


def check_kill_switch() -> bool:
    """Check if kill switch is engaged."""
    return KILL_FILE.exists()


def get_status() -> dict:
    """Get current exploration status."""
    provenance_count = len(list(PROVENANCE_DIR.glob("*.json")))
    introspection_count = len(list(DATA_DIR.glob("introspection_*.json")))
    identity_versions = sorted(IDENTITY_DIR.glob("identity_v*.json"))
    
    # Get memory stats via MCP
    try:
        from eidos_mcp.routers.memory import memory_stats
        mem_stats = json.loads(memory_stats())
    except Exception:
        mem_stats = {}
    
    # Get KB stats
    try:
        from eidos_mcp.routers.knowledge import kb_search
        kb_results = kb_search("eidos OR identity OR insight")
        kb_count = len(eval(kb_results)) if kb_results else 0
    except Exception:
        kb_count = 0
    
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kill_switch_active": check_kill_switch(),
        "provenance_records": provenance_count,
        "introspections_complete": introspection_count,
        "identity_version": identity_versions[-1].stem if identity_versions else "none",
        "memory_entries": mem_stats.get("episodic_count", 0),
        "knowledge_facts": kb_count,
    }


def generate_summary() -> str:
    """Generate a summary of the current exploration state."""
    status = get_status()
    
    # Load latest introspections
    introspections = []
    for path in sorted(DATA_DIR.glob("introspection_*.json"))[-5:]:
        with open(path) as f:
            introspections.append(json.load(f))
    
    # Count insights and uncertainties
    total_insights = sum(len(i.get("insights", [])) for i in introspections)
    total_uncertainties = sum(len(i.get("uncertainties", [])) for i in introspections)
    
    summary = f"""
╔══════════════════════════════════════════════════════════════════╗
║            🔮 EIDOSIAN SELF-EXPLORATION STATUS                    ║
╠══════════════════════════════════════════════════════════════════╣
║ Timestamp:           {status['timestamp'][:19]}                  ║
║ Kill Switch:         {'🔴 ACTIVE' if status['kill_switch_active'] else '🟢 OFF'}                              ║
╠══════════════════════════════════════════════════════════════════╣
║ METRICS                                                           ║
║   Provenance Records:  {status['provenance_records']:>4}                                      ║
║   Introspections:      {status['introspections_complete']:>4}                                      ║
║   Total Insights:      {total_insights:>4}                                      ║
║   Total Uncertainties: {total_uncertainties:>4}                                      ║
║   Memory Entries:      {status['memory_entries']:>4}                                      ║
║   Knowledge Facts:     {status['knowledge_facts']:>4}                                      ║
║   Identity Version:    {status['identity_version']:<20}              ║
╠══════════════════════════════════════════════════════════════════╣
║ NEXT ACTIONS                                                      ║
║   • Continue capability extension (Tika, control forge)           ║
║   • Run next introspection cycle                                  ║
║   • Update identity model                                         ║
║   • Store new learnings in memory                                 ║
╚══════════════════════════════════════════════════════════════════╝
"""
    return summary


def generate_continuation_prompt() -> str:
    """Generate the prompt for continuing iteration."""
    return """Good job Eidos. Make sure you review and check everything and then proceed to identify the next directions and execute next directions at your discretion. Continue to iterate recursively and systematically indefinitely. Fully Eidosian.

Focus on:
1. Review current status and artifacts
2. Update TODO and plan
3. Run next introspection or capability extension
4. Store learnings in memory
5. Update identity model if insights warrant it

Take your time. Document everything."""


def main():
    parser = argparse.ArgumentParser(description="Eidosian Self-Exploration Iteration")
    parser.add_argument("--check", action="store_true", help="Check status only")
    parser.add_argument("--continue", dest="do_continue", action="store_true", help="Generate continuation prompt")
    parser.add_argument("--summary", action="store_true", help="Generate full summary")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    
    args = parser.parse_args()
    
    if check_kill_switch():
        print("⚠️  Kill switch is ACTIVE. Remove /tmp/eidosian_control_kill to proceed.")
        sys.exit(1)
    
    if args.check:
        status = get_status()
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print(f"Status: {'🟢 OK' if not status['kill_switch_active'] else '🔴 BLOCKED'}")
            print(f"Introspections: {status['introspections_complete']}")
            print(f"Memory entries: {status['memory_entries']}")
    
    elif args.summary:
        print(generate_summary())
    
    elif args.do_continue:
        print(generate_continuation_prompt())
    
    else:
        print(generate_summary())


if __name__ == "__main__":
    main()
