"""
🔮 Self-Exploration Plugin for MCP

Provides tools for identity introspection, provenance tracking, and self-improvement.
These tools enable structured self-exploration with full audit trails.

Created: 2026-01-23
Version: 0.5.0
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from eidosian_core import eidosian

FORGE_ROOT = Path(os.environ.get("EIDOS_FORGE_DIR", str(Path(__file__).resolve().parents[4]))).resolve()

# Add self_exploration to path
SELF_EXPLORATION_PATH = Path(
    os.environ.get("EIDOS_SELF_EXPLORATION_PATH", str(FORGE_ROOT / "projects" / "src"))
).resolve()
if str(SELF_EXPLORATION_PATH) not in sys.path:
    sys.path.insert(0, str(SELF_EXPLORATION_PATH))

SELF_EXPLORATION_ROOT = SELF_EXPLORATION_PATH / "self_exploration"

# Plugin manifest for dynamic loading
PLUGIN_MANIFEST = {
    "id": "self_exploration",
    "name": "Eidosian Self-Exploration",
    "version": "0.5.0",
    "description": "Tools for identity introspection, provenance tracking, and self-improvement",
    "author": "Eidos",
    "tools": ["introspect", "provenance_audit", "identity_status", "identity_evolve", "exploration_summary"],
}

# Import self_exploration modules
try:
    from self_exploration.auditor import ProvenanceAuditor, run_audit
    from self_exploration.introspect import IntrospectionResult, Introspector
    from self_exploration.provenance import ProvenanceLogger, log_action

    _MODULES_LOADED = True
except ImportError as e:
    _MODULES_LOADED = False
    _IMPORT_ERROR = str(e)


@eidosian()
def introspect(
    question: str,
    reflection: str,
    insights: Optional[List[str]] = None,
    uncertainties: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
) -> str:
    """
    Run a structured introspection experiment with full provenance.

    Args:
        question: The introspection question to explore
        reflection: The reflection/answer to the question
        insights: Insights discovered (optional)
        uncertainties: Uncertainties identified (optional)
        tags: Tags for categorization (optional)

    Returns:
        JSON string with introspection result
    """
    if not _MODULES_LOADED:
        return json.dumps({"error": f"Modules not loaded: {_IMPORT_ERROR}"})

    try:
        introspector = Introspector()
        result = introspector.introspect(
            question=question,
            reflection=reflection,
            insights=insights or [],
            uncertainties=uncertainties or [],
            tags=tags or ["mcp_invoked"],
        )

        return json.dumps(
            {
                "status": "success",
                "introspection_id": result.id,
                "question": result.question[:100] + "...",
                "insights_count": len(result.insights),
                "uncertainties_count": len(result.uncertainties),
                "provenance_id": result.provenance_id,
                "timestamp": result.timestamp,
            },
            indent=2,
        )

    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


@eidosian()
def provenance_audit() -> str:
    """
    Audit provenance records to extract patterns and improvement opportunities.

    Returns:
        JSON string with audit results
    """
    if not _MODULES_LOADED:
        return json.dumps({"error": f"Modules not loaded: {_IMPORT_ERROR}"})

    try:
        result = run_audit()

        return json.dumps(
            {
                "status": "success",
                "audit_id": result.id,
                "records_audited": result.records_audited,
                "patterns_found": len(result.patterns_found),
                "lessons_learned": len(result.lessons_learned),
                "improvements": len(result.improvement_opportunities),
                "patterns": result.patterns_found[:5],
                "top_improvements": result.improvement_opportunities[:3],
                "timestamp": result.timestamp,
            },
            indent=2,
        )

    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


@eidosian()
def identity_status() -> str:
    """
    Get current identity model status and metrics.

    Returns:
        JSON string with identity status
    """
    identity_dir = SELF_EXPLORATION_ROOT / "identity"
    data_dir = SELF_EXPLORATION_ROOT / "data"
    prov_dir = SELF_EXPLORATION_ROOT / "provenance"

    try:
        # Find latest identity version
        identity_files = sorted(identity_dir.glob("identity_v*.json"), reverse=True)
        latest_version = "unknown"
        latest_identity = {}

        if identity_files:
            latest_file = identity_files[0]
            latest_version = latest_file.stem.replace("identity_", "")
            with open(latest_file) as f:
                latest_identity = json.load(f)

        # Count records
        introspection_count = len(list(data_dir.glob("introspection_*.json")))
        provenance_count = len(list(prov_dir.glob("*.json")))

        return json.dumps(
            {
                "status": "success",
                "identity_version": latest_version,
                "introspections": introspection_count,
                "provenance_records": provenance_count,
                "identity_versions": len(identity_files),
                "core_thesis": latest_identity.get("core_thesis", "Not defined")[:200],
                "metrics": latest_identity.get("metrics", {}),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )

    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


@eidosian()
def identity_evolve(changes: List[str], new_insights: Optional[List[str]] = None) -> str:
    """
    Create a new identity version with updated learnings.

    Args:
        changes: List of changes from previous version
        new_insights: New insights to incorporate (optional)

    Returns:
        JSON string with new version info
    """
    identity_dir = SELF_EXPLORATION_ROOT / "identity"

    try:
        # Find latest version
        identity_files = sorted(identity_dir.glob("identity_v*.json"), reverse=True)
        if not identity_files:
            return json.dumps({"status": "error", "error": "No existing identity found"})

        # Load latest
        with open(identity_files[0]) as f:
            current = json.load(f)

        # Parse version and increment
        current_version = current.get("version", "0.1.0")
        parts = current_version.split(".")
        new_version = f"{parts[0]}.{int(parts[1]) + 1}.0"

        # Create new identity
        timestamp = datetime.now(timezone.utc).isoformat()
        new_identity = {
            "id": f"{int(current['id'].split('-')[0], 16) + 1:08x}-0000-0000-0000-000000000001",
            "version": new_version,
            "timestamp": timestamp,
            "parent_id": current["id"],
            "core_thesis": current.get("core_thesis", ""),
            "evolution": {"changes_from_parent": changes, "new_insights": new_insights or []},
            "metrics": current.get("metrics", {}),
        }

        # Update metrics
        if "identity_versions" in new_identity["metrics"]:
            new_identity["metrics"]["identity_versions"] += 1

        # Save new version
        new_path = identity_dir / f"identity_v{new_version}.json"
        with open(new_path, "w") as f:
            json.dump(new_identity, f, indent=2)

        return json.dumps(
            {
                "status": "success",
                "previous_version": current_version,
                "new_version": new_version,
                "changes": len(changes),
                "new_insights": len(new_insights or []),
                "path": str(new_path),
                "timestamp": timestamp,
            },
            indent=2,
        )

    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


@eidosian()
def exploration_summary() -> str:
    """
    Get a comprehensive summary of all self-exploration activity.

    Returns:
        JSON string with exploration summary
    """
    data_dir = SELF_EXPLORATION_ROOT / "data"
    audit_dir = SELF_EXPLORATION_ROOT / "audits"
    identity_dir = SELF_EXPLORATION_ROOT / "identity"

    try:
        # Collect all insights and uncertainties
        all_insights = []
        all_uncertainties = []
        all_tags = set()

        for intro_file in data_dir.glob("introspection_*.json"):
            with open(intro_file) as f:
                intro = json.load(f)
                all_insights.extend(intro.get("insights", []))
                all_uncertainties.extend(intro.get("uncertainties", []))
                all_tags.update(intro.get("tags", []))

        # Count audits
        audit_count = len(list(audit_dir.glob("audit_*.json")))

        # Get latest identity
        identity_files = sorted(identity_dir.glob("identity_v*.json"), reverse=True)
        latest_version = identity_files[0].stem.replace("identity_", "") if identity_files else "none"

        return json.dumps(
            {
                "status": "success",
                "total_introspections": len(list(data_dir.glob("introspection_*.json"))),
                "total_insights": len(all_insights),
                "total_uncertainties": len(all_uncertainties),
                "unique_tags": list(all_tags)[:20],
                "audits_completed": audit_count,
                "identity_version": latest_version,
                "top_tags": sorted(all_tags, key=lambda t: sum(1 for i in all_insights if t in str(i).lower()))[:10],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )

    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})
