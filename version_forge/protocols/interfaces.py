"""
Type protocols defining interfaces for version components.
These contracts ensure structural typing compatibility across modules.
"""
from typing import Any, Dict, List, Literal, Optional, Protocol, TypeVar, Union

# Try to import TypedDict from typing (Python 3.8+)
try:
    from typing import TypedDict
except ImportError:
    # Fall back to typing_extensions for older Python versions
    from typing_extensions import TypedDict

# Type variables for generic type constraints
T = TypeVar('T')
VersionT = TypeVar('VersionT', bound='VersionProtocol')

ConfigSource = Literal["environment", "local_file", "module", "central_db", "defaults"]

class VersionDict(TypedDict, total=False):
    """Version data with structural completeness"""
    version: str
    major: int
    minor: int
    patch: int
    release_date: str
    min_version: str
    source: ConfigSource

class VersionDelta(TypedDict, total=False):
    """Semantic version difference metrics"""
    major: int
    minor: int
    patch: int
    is_upgrade: bool
    is_downgrade: bool
    is_same: bool
    error: str

class VersionProtocol(Protocol):
    """Contract defining version behavior"""
    major: int
    minor: int
    patch: int
    def __lt__(self, other: Any) -> bool: ...
    def __eq__(self, other: Any) -> bool: ...
    def __str__(self) -> str: ...

class IVersioned(Protocol):
    """Contract for version-aware components"""
    @property
    def version(self) -> VersionProtocol: ...

    @property
    def version_info(self) -> Dict[str, Any]: ...

# Type aliases for clarity and consistency
VersionData = VersionDict

class SourceLoader(Protocol):
    def __call__(self) -> Optional[VersionData]: ...

VersionLike = Union[str, VersionProtocol, Any]  # Any for external version types

class VersionUpdateResult(TypedDict):
    """Version update operation metrics"""
    updated: bool
    files_changed: List[str]
    files_examined: int
    duration_seconds: float
    version_from: str
class VersionStatus(TypedDict):
    """Comprehensive version context"""
    version: str
    components: Dict[str, int]
    release_date: str
    min_version: str
    version_source: str
    module_path: str
    system_info: Dict[str, str]
    module_path: str
    system_info: Dict[str, str]
