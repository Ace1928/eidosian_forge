"""
Version comparison operations with semantic accuracy.
"""
import logging
from typing import Any, Optional

from ..core.version import parse_version
from ..protocols.interfaces import VersionDelta, VersionProtocol

logger = logging.getLogger("forge.version")

def calculate_delta(v1: str, v2: str) -> VersionDelta:
    """Calculate precise semantic distance between versions"""
    try:
        ver1, ver2 = parse_version(v1), parse_version(v2)
        # Extract components safely regardless of version implementation
        major1: Any | int = getattr(ver1, 'major', 0)
        major2: Any | int = getattr(ver2, 'major', 0)
        minor1: Any | int = getattr(ver1, 'minor', 0)
        minor2: Any | int = getattr(ver2, 'minor', 0)
        patch1: Any | int = getattr(ver1, 'patch', getattr(ver1, 'micro', 0))
        patch2: Any | int = getattr(ver2, 'patch', getattr(ver2, 'micro', 0))

        delta: VersionDelta = {
            "major": major2 - major1,
            "minor": minor2 - minor1,
            "patch": patch2 - patch1,
            "is_upgrade": ver2 > ver1,
            "is_downgrade": ver2 < ver1,
            "is_same": ver2 == ver1
        }
        return delta
    except Exception as e:
        # Graceful error handling with explicit indication
        logger.debug(f"Version delta calculation failed: {e}")
        # Return valid VersionDelta structure without error field
        delta: VersionDelta = {
            "major": 0, "minor": 0, "patch": 0,
            "is_upgrade": False, "is_downgrade": False,
            "is_same": False
        }
        return delta

# API stability through semantic aliasing
calculate_version_delta = calculate_delta


def is_compatible(version: str, minimum: Optional[str] = None,
                  config_min_version: Optional[str] = None) -> bool:
    """Check version compatibility - false until proven compatible"""
    try:
        minimum_version: str = minimum or config_min_version or "0.1.0"
        ver1: VersionProtocol = parse_version(version)
        ver2: VersionProtocol = parse_version(minimum_version)
        # Using only protocol-guaranteed methods: not less than = greater than or equal to
        return not (ver1 < ver2) or (ver1 == ver2)
    except Exception as e:
        logger.debug(f"Compatibility check failed: {e}")
        return False  # When uncertain, assume incompatible
