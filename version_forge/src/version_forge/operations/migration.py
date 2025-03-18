"""
Version migration guide generation with semantic understanding.
"""
import logging
from typing import Any, Dict, List, TypedDict

from ..operations.compare import calculate_delta
from ..protocols.interfaces import VersionDelta

logger = logging.getLogger("forge.version")

class MigrationGuide(TypedDict):
    """Type definition for migration guide structure."""
    component: str
    from_version: str
    to_version: str
    version_delta: VersionDelta
    upgrade_type: str
    estimated_effort: str
    breaking_changes: List[str]
    new_features: List[str]
    deprecations: List[str]
    suggestions: List[str]

class MigrationGuideGenerator:
    """Generate migration guides between versions with automatic suggestions"""

    def __init__(self) -> None:
        self._known_migrations: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def register_migration_info(self, component: str, from_version: str, to_version: str,
                              breaking_changes: List[str],
                              new_features: List[str],
                              deprecations: List[str]) -> None:
        """Register known migration information for a specific version transition"""
        key = f"{from_version}_to_{to_version}"

        # Initialize component dict if it doesn't exist
        if component not in self._known_migrations:
            self._known_migrations[component] = {}

        # Store migration information
        self._known_migrations[component][key] = {
            "breaking_changes": breaking_changes,
            "new_features": new_features,
            "deprecations": deprecations
        }

    def generate_migration_guide(self, component: str, from_version: str,
                               to_version: str) -> MigrationGuide:
        """Generate a migration guide between versions"""
        # Calculate version difference
        delta = calculate_delta(from_version, to_version)

        # Look up any known migration information
        key = f"{from_version}_to_{to_version}"
        known_info = self._known_migrations.get(component, {}).get(key, {})

        # Build migration guide with explicit type annotation
        guide: MigrationGuide = {
            "component": component,
            "from_version": from_version,
            "to_version": to_version,
            "version_delta": delta,
            "upgrade_type": self._determine_upgrade_type(delta),
            "estimated_effort": self._estimate_effort(delta, known_info),
            "breaking_changes": known_info.get("breaking_changes", []),
            "new_features": known_info.get("new_features", []),
            "deprecations": known_info.get("deprecations", []),
            "suggestions": self._generate_suggestions(component, delta, known_info),
        }

        return guide

    def _determine_upgrade_type(self, delta: VersionDelta) -> str:
        """Classify the type of upgrade based on semantic versioning rules"""
        if not delta.get("is_upgrade", False):
            if delta.get("is_same", False):
                return "no_change"
            return "downgrade"

        if delta.get("major", 0) > 0:
            return "major_upgrade"
        elif delta.get("minor", 0) > 0:
            return "minor_upgrade"
        elif delta.get("patch", 0) > 0:
            return "patch_upgrade"
        return "unknown"

    def _estimate_effort(self, delta: VersionDelta, known_info: Dict[str, Any]) -> str:
        """Estimate the effort required for migration"""
        # If we have specific information, use that
        if known_info:
            breaking_count = len(known_info.get("breaking_changes", []))
            if breaking_count > 5:
                return "high"
            elif breaking_count > 0:
                return "medium"
            return "low"

        # Otherwise estimate based on version numbers
        if delta.get("major", 0) > 1:
            return "very_high"
        elif delta.get("major", 0) == 1:
            return "high"
        elif delta.get("minor", 0) > 5:
            return "medium_high"
        elif delta.get("minor", 0) > 0:
            return "medium"
        return "low"

    def _generate_suggestions(self, component: str,
                            delta: VersionDelta,
                            known_info: Dict[str, Any]) -> List[str]:
        """Generate helpful migration suggestions"""
        suggestions: List[str] = []

        # Basic suggestions based on semantic versioning
        upgrade_type = self._determine_upgrade_type(delta)

        # Component-specific suggestions based on component type
        if component.lower().startswith("api"):
            suggestions.append(f"Check for API endpoint changes in {component}")
        elif component.lower().startswith("ui"):
            suggestions.append(f"Review UI component changes in {component}")
        elif component.lower().startswith("core"):
            suggestions.append(f"Test core functionality affected by {component} changes")

        # General suggestions based on upgrade type
        if upgrade_type == "major_upgrade":
            suggestions.append("Review all APIs for breaking changes")
            suggestions.append("Update tests to account for new behaviors")
            suggestions.append("Consider a phased migration approach")
        elif upgrade_type == "minor_upgrade":
            suggestions.append("Check documentation for new features")
            suggestions.append("Look for deprecated features you might be using")
        elif upgrade_type == "patch_upgrade":
            suggestions.append("Review bug fixes to see if they impact your usage")

        # Add specific suggestions if we have known info
        if known_info and known_info.get("breaking_changes"):
            suggestions.append("Address all breaking changes listed above")

        return suggestions
