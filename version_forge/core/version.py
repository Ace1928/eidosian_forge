"""
Core version parsing and manipulation functionality.
Provides robust semantic version handling with flexible parsing strategies.
"""
import logging
import re
from typing import Any, Optional

from ..protocols.interfaces import VersionProtocol

logger = logging.getLogger("forge.version")

class SimpleVersion:
    """Semantic version with numerical precision and lexical comparison"""
    __slots__ = ("major", "minor", "patch", "micro", "prerelease", "_valid")

    def __init__(self, version_str: str) -> None:
        self.major: int = 0
        self.minor: int = 0
        self.patch: int = 0
        self.micro: int = 0  # Alias for patch to maintain API compatibility
        self.prerelease: Optional[str] = None
        self._valid: bool = self._parse(version_str)

    def _parse(self, version_str: str) -> bool:
        """Parse version string with component extraction"""
        if match := re.match(r'^(\d+)\.(\d+)\.(\d+)(?:[-.]?(.+))?$', version_str):
            self.major, self.minor = int(match.group(1)), int(match.group(2))
            self.patch = self.micro = int(match.group(3))  # synchronize patch/micro
            self.prerelease = match.group(4) or None
            return True
        return False

    def __lt__(self, other: Any) -> bool:
        """Compare versions: 1.0.0 < 2.0.0 and 1.0.0 > 1.0.0-alpha"""
        if not isinstance(other, SimpleVersion):
            return NotImplemented

        # Short-circuit numeric comparisons for efficiency
        for s, o in zip((self.major, self.minor, self.patch),
                       (other.major, other.minor, other.patch)):
            if s != o:
                return s < o

        # Prerelease logic: absent > present, or lexical comparison
        if self.prerelease is None:
            return False  # No prerelease is "greater than" any prerelease
        if other.prerelease is None:
            return True   # Any prerelease is "less than" no prerelease
        return self.prerelease < other.prerelease  # Lexical comparison

    def __eq__(self, other: Any) -> bool:
        """Versions equal when all components match exactly"""
        if not isinstance(other, SimpleVersion):
            return NotImplemented
        return ((self.major, self.minor, self.patch, self.prerelease) ==
                (other.major, other.minor, other.patch, other.prerelease))

    # Derived comparison operators through mathematical composition
    def __gt__(self, other: Any) -> bool:
        lt_result = self.__lt__(other)
        return NotImplemented if lt_result is NotImplemented else not (lt_result or self.__eq__(other))

    def __ge__(self, other: Any) -> bool:
        lt_result = self.__lt__(other)
        return NotImplemented if lt_result is NotImplemented else not lt_result

    def __le__(self, other: Any) -> bool:
        """Less than or equal to comparison"""
        eq_result = self.__eq__(other)
        if eq_result is NotImplemented:
            return NotImplemented
        if eq_result:
            return True
        return self.__lt__(other)

    def __str__(self) -> str:
        """String representation"""
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        return version

    def __repr__(self) -> str:
        """Unambiguous developer representation"""
        return f"SimpleVersion({self.__str__()})"

    def __hash__(self) -> int:
        """Hash based on version components for dictionary use"""
        return hash((self.major, self.minor, self.patch, self.prerelease))


def parse_version(version_str: str, fallback_to_simple: bool = True) -> VersionProtocol:
    """Parse version string with optimal strategy selection"""
    # Normalize input for consistent results
    cleaned: str = re.sub(r'^[vV]', '', version_str.strip())

    # Strategy cascade with graceful degradation
    try:
        # Preferred implementation when available
        from packaging.version import parse as packaging_parse

        # Create adapter to ensure protocol compliance
        pkg_ver = packaging_parse(cleaned)
        if hasattr(pkg_ver, 'release'):
            # Convert packaging.Version to SimpleVersion for protocol compliance
            release = pkg_ver.release
            ver = SimpleVersion(f"{release[0]}.{release[1]}.{release[2] if len(release) > 2 else 0}")
            # Copy additional attributes for compatibility
            if hasattr(pkg_ver, 'pre') and pkg_ver.pre:
                ver.prerelease = '.'.join(str(x) for x in pkg_ver.pre)
            return ver

        # Fallback for unexpected version types
        return SimpleVersion(cleaned)
    except ImportError:
        # Self-contained fallback requires no external dependencies
        return SimpleVersion(cleaned)
    except Exception as e:
        # Controlled failure with explicit instruction
        if not fallback_to_simple:
            raise
        logger.debug(f"Standard parser failed: {e}")
        return SimpleVersion(cleaned)


DEFAULT_VERSION = "0.1.0"


def format_version(version: Any) -> str:
    """Convert any version to canonical form with v-prefix"""
    try:
        if isinstance(version, str):
            return f"v{version.lstrip('vV')}"
        elif hasattr(version, '__version__'):
            return f"v{getattr(version, '__version__').lstrip('vV')}"
        elif hasattr(version, 'version'):
            return f"v{getattr(version, 'version').lstrip('vV')}"
        else:
            return f"v{str(version).lstrip('vV')}"
    except Exception as e:
        logger.debug(f"Format error: {e}")
        return f"v{DEFAULT_VERSION}"
