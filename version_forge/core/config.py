"""
Version configuration handling with robust defaults and synchronization.
"""
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Final

from ..protocols.interfaces import ConfigSource

# Constants with semantically pure defaults
DEFAULT_VERSION: Final[str] = "0.1.0"
DEFAULT_MIN_VERSION: Final[str] = "0.1.0"
DEFAULT_RELEASE_DATE: Final[str] = datetime.now().strftime("%Y-%m-%d")

# Environment-aware paths
EIDOSIAN_ROOT: Final[Path] = Path(os.environ.get(
    "EIDOSIAN_ROOT", str(Path.home() / "repos")
))
CENTRAL_VERSIONS_PATH: Final[Path] = Path(os.environ.get(
    "CENTRAL_VERSIONS_PATH", str(EIDOSIAN_ROOT / "central_versions.json")
))

@dataclass
class VersionConfig:
    """Version configuration with bidirectional synchronization"""
    __version__: str = DEFAULT_VERSION
    min_version: str = DEFAULT_MIN_VERSION
    release_date: str = DEFAULT_RELEASE_DATE
    major: int = field(default=0, init=False)
    minor: int = field(default=1, init=False)
    patch: int = field(default=0, init=False)
    source: ConfigSource = "defaults"

    def __post_init__(self) -> None:
        """Initialize computed fields after direct initialization"""
        self._sync_from_version()

    def _sync_from_version(self) -> None:
        """Extract semantic components from version string"""
        if match := re.match(r'^(\d+)\.(\d+)\.(\d+)', self.__version__):
            self.major, self.minor, self.patch = map(int, match.groups())

    def update(self, **kwargs: Any) -> None:
        """Update with bidirectional component synchronization"""
        # Apply valid updates to attributes
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        # Keep string and numeric representations in perfect sync
        if "__version__" in kwargs:
            self._sync_from_version()
        elif any(k in kwargs for k in ("major", "minor", "patch")):
            self.__version__ = f"{self.major}.{self.minor}.{self.patch}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "version": self.__version__,
            "min_version": self.min_version,
            "release_date": self.release_date,
            "major": self.major,
            "minor": self.minor,
            "patch": self.patch,
            "source": self.source
        }
