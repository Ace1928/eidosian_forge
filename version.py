# 🎯 Version - Precision through elegance 🚀
"""
Universal Version Module
------------------------
Configuration cascade with deterministic priority:
1. Environment (PACKAGE_FORGE_VERSION) - explicit overrides
2. Local files (.version.json) - project-specific truth
3. Modules (config.py, __init__.py) - code-embedded reality
4. Central DB (central_versions.json) - organizational cohesion
5. Defaults (last resort) - graceful degradation
"""
import os
import re
import json
import logging
import inspect
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Final, List, Literal, Optional, Protocol, Tuple, TypeVar, Union, cast, TypedDict
from functools import lru_cache
from dataclasses import dataclass, field

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 📐 Type Definitions - Structural contracts as guarantees
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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

# Function signatures with precise intent
VersionData = VersionDict
SourceLoader = Callable[[], Optional[VersionData]]
VersionLike = Union[str, 'SimpleVersion', Any]  # Any for external version types

class VersionProtocol(Protocol):
    """Contract defining version behavior"""
    major: int
    minor: int
    patch: int
    def __lt__(self, other: Any) -> bool: ...
    def __eq__(self, other: Any) -> bool: ...

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 📝 Logging - Self-aware observability
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

logger = logging.getLogger("forge.version")
debug_enabled: Final[bool] = os.environ.get("FORGE_DEBUG") == "1"
repo_debug_vars: Final[List[str]] = [
    var for var in os.environ
    if var.endswith("_FORGE_DEBUG") and os.environ[var] == "1"
]

if debug_enabled or repo_debug_vars:
    logging.basicConfig(level=logging.DEBUG)
    logger.setLevel(logging.DEBUG)
    logger.debug("📝 Debug activated - quantum observability engaged")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🧩 Repository Detection - Layered fallback strategies
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass(frozen=True)
class RepoInfo:
    """Immutable repository identification"""
    name: str
    path: Path
    package: str

def normalize_repo_name(name: str) -> str:
    """Normalize repository name to forge standard"""
    return f"{name.removesuffix('_repo').removesuffix('_forge')}_forge"

@lru_cache(maxsize=1)
def detect_forge_repo(search_levels: int = 3) -> RepoInfo:
    """Repository detection with multi-strategy fallback"""
    def check_path(path: Path) -> Optional[RepoInfo]:
        if path.name.endswith(('_forge', '_forge_repo')):
            name = normalize_repo_name(path.name)
            return RepoInfo(name, path, name.replace('_forge', ''))
        return None

    # Strategies in reverse fallback order - most specific first
    strategies: Dict[str, Callable[[], Optional[RepoInfo]]] = {
        # Explicit environment override - user has declared intent
        "env": lambda: RepoInfo(
            normalize_repo_name(env_name),
            Path(os.environ.get("FORGE_REPO_PATH", str(Path.cwd()))),
            normalize_repo_name(env_name).replace('_forge', '')
        ) if (env_name := os.environ.get("FORGE_REPO_NAME")) else None,

        # Stack frame analysis - caller's context reveals identity
        "frame": lambda: next((
            check_path(Path(frame.filename).resolve().parents[level])
            for frame in inspect.stack()
            if frame.filename != __file__
            for level in range(search_levels + 1)
        ), None),

        # Working directory context - location implies identity
        "cwd": lambda: next((
            check_path(Path.cwd().parents[level] if level else Path.cwd())
            for level in range(search_levels + 1)
        ), None),

        # Module location - file system placement reveals purpose
        "module": lambda: next((
            check_path(path) for path in (
                Path(__file__).resolve().parent,
                Path(__file__).resolve().parent.parent
            )
        ), None),
    }

    # Try each strategy in priority order
    for name, strategy in strategies.items():
        try:
            if result := strategy():
                logger.debug(f"Repository detected via {name} strategy")
                return result
        except Exception as e:
            logger.debug(f"{name} detection failed: {e}")

    # Controlled uncertainty fallback with useful defaults
    logger.warning("⚠️ Repository detection failed: using generic identity")
    return RepoInfo("generic_forge", Path.cwd(), "generic")

# Singleton repository info - calculated once, used everywhere
REPO: Final[RepoInfo] = detect_forge_repo()
logger.debug(f"Repo: {REPO.name} at {REPO.path}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ⚙️ Configuration - Constants with semantically pure defaults
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEFAULT_VERSION: Final[str] = "0.1.0"
DEFAULT_MIN_VERSION: Final[str] = "0.1.0"
DEFAULT_RELEASE_DATE: Final[str] = datetime.now().strftime("%Y-%m-%d")

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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🔍 Version Source Detection - Strategic cascading fallbacks
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def find_local_version_file() -> Optional[Path]:
    """Find version file using prioritized naming patterns"""
    candidates = (
        f"{REPO.package}_version.json",
        "version.json",
        ".version.json",
        f"{REPO.name}.version",
        "version.cfg"
    )
    return next((p for p in (REPO.path / name for name in candidates) if p.exists()), None)

def load_from_source(loader: SourceLoader, source_name: str) -> Optional[VersionData]:
    """Load from source with error handling and logging"""
    try:
        if data := loader():
            logger.debug(f"✓ Version from {source_name}: {data.get('version', '?')}")
            return data
    except Exception as e:
        logger.debug(f"Source {source_name} failed: {e}")
    return None

def load_from_module(module_name: str) -> Optional[VersionData]:
    """Extract version from Python module attributes"""
    try:
        module = __import__(module_name)

        # Find first available version attribute
        version = next((
            getattr(module, attr) for attr in ("VERSION", "__version__", "version")
            if hasattr(module, attr)
        ), None)

        if not version:
            return None

        # Map module attributes to standardized keys with semantic precision
        attr_mapping: Dict[str, List[str]] = {
            "major": ["VERSION_MAJOR", "MAJOR"],
            "minor": ["VERSION_MINOR", "MINOR"],
            "patch": ["VERSION_PATCH", "PATCH"],
            "release_date": ["VERSION_RELEASE_DATE", "RELEASE_DATE"],
            "min_version": ["MINIMUM_VERSION", "MIN_VERSION"]
        }

        # Build result with first matching attributes - efficiency through priority
        result: VersionData = {"version": version, "source": "module"}
        for result_key, attr_options in attr_mapping.items():
            for attr in attr_options:
                if hasattr(module, attr):
                    result[result_key] = getattr(module, attr)
                    break

        return result
    except (ImportError, AttributeError):
        return None

def load_from_central_db() -> Optional[VersionData]:
    """Query central version registry for organizational coherence"""
    if not CENTRAL_VERSIONS_PATH.exists():
        return None

    try:
        with open(CENTRAL_VERSIONS_PATH, "r", encoding="utf-8") as f:
            db_data = json.load(f).get(REPO.name)
            if db_data:
                # Cast the merged dictionary to VersionData type for type safety
                return cast(VersionData, {"source": "central_db", **db_data})
    except Exception as e:
        logger.debug(f"Central DB error: {e}")

    return None

def load_from_local_file() -> Optional[VersionData]:
    """Extract version from local config files"""
    if not (file_path := find_local_version_file()):
        return None

    try:
        # Parse based on file extension - adapt to format
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f) if file_path.suffix == '.json' else {
                k.strip(): v.strip().strip('"\'')
                for line in f
                if line.strip() and not line.startswith('#')
                for k, v in (line.split('=', 1),)
            }

        # Map to standardized keys - translation table for semantic consistency
        key_map: Dict[str, str] = {
            "version": "version",
            "__version__": "version",
            "major": "major",
            "minor": "minor",
            "patch": "patch",
            "release_date": "release_date",
            "min_version": "min_version"
        }

        result = {
            std_key: data[file_key]
            for file_key, std_key in key_map.items()
            if file_key in data
        }

        return cast(VersionData, {"source": "local_file", **result}) if "version" in result else None
    except Exception as e:
        logger.debug(f"Local file error: {e}")

    return None

def load_from_env_vars() -> Optional[VersionData]:
    """Extract version from environment variables for explicit control"""
    prefix: Final[str] = f"{REPO.package.upper()}_FORGE"

    # Type-safe environment variable mapping with conversion functions
    env_vars: Dict[str, Tuple[str, Callable[[str], Any]]] = {
        "version": (f"{prefix}_VERSION", str),
        "major": (f"{prefix}_MAJOR", int),
        "minor": (f"{prefix}_MINOR", int),
        "patch": (f"{prefix}_PATCH", int),
        "release_date": (f"{prefix}_RELEASE_DATE", str),
        "min_version": (f"{prefix}_MIN_VERSION", str)
    }

    result: VersionData = {"source": "environment"}
    for key, (var_name, converter) in env_vars.items():
        if value := os.environ.get(var_name):
            try:
                result[key] = converter(value)
            except ValueError:
                logger.debug(f"Cannot convert {var_name}={value!r}")

    return result if "version" in result else None

@lru_cache(maxsize=1)
def load_version_config() -> VersionConfig:
    """Load version using priority-based source cascade with fallbacks"""
    config = VersionConfig()

    # Sources in strict priority order - explicit to implicit
    sources: List[Tuple[SourceLoader, str]] = [
        (load_from_env_vars, "environment"),
        (load_from_local_file, "local file"),
        (lambda: load_from_module(f"{REPO.package}_forge.config"), "config module"),
        (lambda: load_from_module(f"{REPO.package}_forge"), "package module"),
        (lambda: load_from_module(REPO.package), "base module"),
        (load_from_central_db, "central database")
    ]

    # First valid source wins - early decision circuit
    for loader, name in sources:
        if data := load_from_source(loader, name):
            if "version" in data:
                source_type = cast(ConfigSource, data.pop("source", "unknown"))
                config.update(**data)
                config.source = source_type
                break

    return config

# Initialize singleton version configuration - calculate once, use everywhere
version_config: Final[VersionConfig] = load_version_config()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🔢 Version Parsing - Mathematical semantics in structural form
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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
        gt_result = self.__gt__(other)
        return NotImplemented if gt_result is NotImplemented else not gt_result

    def __str__(self) -> str:
        """Canonical representation with error flagging"""
        base = f"{self.major}.{self.minor}.{self.patch}"
        prerelease = f"-{self.prerelease}" if self.prerelease else ""
        warning = " [⚠️]" if not self._valid else ""
        return f"{base}{prerelease}{warning}"

    def __repr__(self) -> str:
        """Unambiguous developer representation"""
        return f"SimpleVersion({str(self)})"

    def parse_version(version_str: str, fallback_to_simple: bool = True) -> VersionProtocol:
        """Parse version string with optimal strategy selection"""
        # Normalize input for consistent results
        cleaned: str = re.sub(r'^[vV]', '', version_str.strip())

        # Strategy cascade with graceful degradation
        try:
            # Preferred implementation when available
            from packaging.version import parse as packaging_parse
            from packaging.version import Version as PackagingVersion

            # Create adapter to ensure protocol compliance
            pkg_ver = packaging_parse(cleaned)
            if isinstance(pkg_ver, PackagingVersion):
                # Convert packaging.Version to SimpleVersion for protocol compliance
                ver = SimpleVersion(f"{pkg_ver.release[0]}.{pkg_ver.release[1]}.{pkg_ver.release[2] if len(pkg_ver.release) > 2 else 0}")
                # Copy additional attributes for compatibility
                if pkg_ver.pre:
                        ver.prerelease = '.'.join(str(x) for x in pkg_ver.pre)
                return ver

            # Fallback for unexpected version types
            return SimpleVersion(cleaned)
        except ImportError:
            # Self-contained fallback requires no external dependencies
                \return SimpleVersion(cleaned)
            except Exception as e:
                # Controlled failure with explicit instruction
                if not fallback_to_simple:
                    raise
                logger.debug(f"Standard parser failed: {e}")
                return SimpleVersion(cleaned)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🔍 Version Operations - Pure transformations with precision
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import sys


def format_version(version: VersionLike) -> str:
    """Convert any version to canonical form with v-prefix"""
    try:
        parsed_version = parse_version(version) if isinstance(version, str) else version
        patch = getattr(parsed_version, 'patch', getattr(parsed_version, 'micro', 0))
        return f"v{parsed_version.major}.{parsed_version.minor}.{patch}"
    except Exception:
        # Last-resort fallback preserves operational continuity
        return f"v{version}"


def is_compatible(version: str, minimum: Optional[str] = None) -> bool:
    """Check version compatibility - false until proven compatible"""
    try:
        minimum_version = minimum or version_config.min_version
        return parse_version(version) >= parse_version(minimum_version)
    except Exception as e:
        logger.debug(f"Compatibility check failed: {e}")
        return False  # When uncertain, assume incompatible


class VersionDelta(TypedDict, total=False):
    """Version comparison results with structural completeness"""
    major: int          # Major version difference
    minor: int          # Minor version difference
    patch: int          # Patch version difference
    is_upgrade: bool    # v1 < v2
    is_downgrade: bool  # v1 > v2
    is_same: bool       # v1 == v2
    error: str          # Optional error message


def calculate_delta(v1: str, v2: str) -> VersionDelta:
    """Calculate precise semantic distance between versions"""
    try:
        ver1, ver2 = parse_version(v1), parse_version(v2)
        # Extract patch regardless of version implementation
        patch1 = getattr(ver1, 'patch', getattr(ver1, 'micro', 0))
        patch2 = getattr(ver2, 'patch', getattr(ver2, 'micro', 0))

        return {
            "major": ver2.major - ver1.major,
            "minor": ver2.minor - ver1.minor,
            "patch": patch2 - patch1,
            "is_upgrade": ver2 > ver1,
            "is_downgrade": ver2 < ver1,
            "is_same": ver2 == ver1
        }
    except Exception as e:
        # Graceful error handling with explicit indication
        return {
            "major": 0, "minor": 0, "patch": 0,
            "is_upgrade": False, "is_downgrade": False,
            "is_same": False, "error": str(e)
        }


# API stability through semantic aliasing
calculate_version_delta = calculate_delta


class VersionUpdateResult(TypedDict):
    """Version update operation metrics"""
    updated: bool               # Whether any files were modified
    files_changed: List[str]    # Paths of modified files
    files_examined: int         # Total files scanned
    duration_seconds: float     # Operation duration
    version_from: str           # Original version
    version_to: str             # Target version


def update_version(new_version: str, repo_path: Optional[Path] = None) -> VersionUpdateResult:
    """Update version references throughout codebase with surgical precision."""
    current_version: str = version_config.__version__
    start_time: datetime = datetime.now()
    repo_root: Path = repo_path or REPO.path

    # Fast path for no-op updates - avoid unnecessary file operations
    if current_version == new_version:
        return {
            "updated": False,
            "files_changed": [],
            "files_examined": 0,
            "duration_seconds": 0.0,
            "version_from": current_version,
            "version_to": new_version
        }

    # Validate version format with fail-fast principle
    if not (version_match := re.match(r'^(\d+)\.(\d+)\.(\d+)', new_version)):
        raise ValueError(f"Invalid version format: {new_version}")

    # Parse components once for efficiency
    new_major, new_minor, new_patch = map(int, version_match.groups())

    files_updated: List[str] = []
    files_examined: int = 0

    # Surgical replacement patterns with capture groups for precise changes
    version_patterns: List[Tuple[str, str]] = [
        (rf'(version\s*=\s*["\']){re.escape(current_version)}(["\'])',
         rf'\g<1>{new_version}\g<2>'),
        (rf'(__version__\s*=\s*["\']){re.escape(current_version)}(["\'])',
         rf'\g<1>{new_version}\g<2>'),
        (rf'(VERSION_MAJOR\s*=\s*){version_config.major}',
         rf'\g<1>{new_major}'),
        (rf'(VERSION_MINOR\s*=\s*){version_config.minor}',
         rf'\g<1>{new_minor}'),
        (rf'(VERSION_PATCH\s*=\s*){version_config.patch}',
         rf'\g<1>{new_patch}'),
    ]

    # Exclusion rules for efficiency and safety
    skip_dirs: set[str] = {
        "__pycache__", "dist", "build", "venv", ".venv",
        ".git", "node_modules"
    }
    valid_extensions: set[str] = {
        ".py", ".md", ".rst", ".txt", ".toml", ".yaml", ".yml", ".cfg"
    }

    def update_file(path: Path) -> bool:
        """Update version references in a file, return True if changed"""
        nonlocal files_examined
        files_examined += 1

        try:
            content: str = path.read_text(encoding="utf-8")
            new_content: str = content

            for pattern, replacement in version_patterns:
                new_content = re.sub(pattern, replacement, new_content)

            # Idempotent write - only modify when necessary
            if new_content != content:
                path.write_text(new_content, encoding="utf-8")
                files_updated.append(str(path.relative_to(repo_root)))
                return True
            return False
        except Exception as e:
            logger.debug(f"Failed updating {path}: {e}")
            return False

    # Process files with efficient filtering
    for file_path in repo_root.rglob('*'):
        # Fast-path exclusions first to minimize work
        if (file_path.is_dir() or
            file_path.name.startswith('.') or
            any(p in file_path.parts for p in skip_dirs)):
            continue

        # Process only relevant file types
        if file_path.suffix.lower() in valid_extensions:
            update_file(file_path)

    # Special handling for critical configuration files
    pyproject_path = repo_root / "pyproject.toml"
    if pyproject_path.exists():
        update_file(pyproject_path)

    # Calculate operation duration for metrics
    duration: float = (datetime.now() - start_time).total_seconds()

    # Update runtime state for consistency
    if files_updated:
        version_config.update(
            __version__=new_version,
            major=new_major, minor=new_minor, patch=new_patch
        )

        # Message with contextually appropriate wit
        log_message = f"Updated {len(files_updated)} files to {new_version}"
        if duration < 1.0:
            logger.info(f"{log_message} faster than a quantum fluctuation ⚡")
        elif duration > 5.0:
            logger.info(f"{log_message} with the inevitability of entropy ♾️")
        else:
            logger.info(f"{log_message} ✓")

        # Detailed logging only when appropriate
        if logger.level <= logging.DEBUG:
            shown = files_updated[:3]
            if len(files_updated) > 3:
                shown.append(f"...and {len(files_updated) - 3} more")
            logger.debug("Modified: " + ", ".join(shown))
    else:
        logger.info("No version references found - perfect stealth or nothing to change")

    return {
        "updated": bool(files_updated),
        "files_changed": files_updated,
        "files_examined": files_examined,
        "duration_seconds": duration,
        "version_from": current_version,
        "version_to": new_version
    }


# API stability through semantic aliasing
update_version_universally = update_version


class VersionStatus(TypedDict):
    """Comprehensive version context"""
    version: str
    components: Dict[str, int]
    release_date: str
    min_version: str
    version_source: str
    module_path: str
    system_info: Dict[str, str]


def get_version_status() -> VersionStatus:
    """Gather complete version diagnostics"""
    return {
        "version": version_config.__version__,
        "components": {
            "major": version_config.major,
            "minor": version_config.minor,
            "patch": version_config.patch
        },
        "release_date": version_config.release_date,
        "min_version": version_config.min_version,
        "version_source": version_config.source,
        "module_path": __file__,
        "system_info": {
            "python_version": sys.version.split()[0],
            "platform": sys.platform,
            "timestamp": datetime.now().isoformat()
        }
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🚀 Command Interface - Structural information manipulation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    from typing import Callable, NoReturn, Protocol

    # ┌────────────────────────────────────────────────────────┐
    # │ Type Definitions - Command execution contracts         │
    # └────────────────────────────────────────────────────────┘
    class CommandHandler(Protocol):
        """Contract for command execution functions"""
        def __call__(self, args: List[str]) -> None: ...

    def display_header() -> None:
        """Present visual identity with consistent styling"""
        print("╔════════════════════════════════════════════════╗")
        print("║  🌀 Version Control Nexus v3.14.7 🌀           ║")
        print("╚════════════════════════════════════════════════╝")

    def print_comparison(v1: str, v2: str) -> None:
        """Generate visual representation of version differences"""
        delta: VersionDelta = calculate_delta(v1, v2)

        print(f"📊 {v1} → {v2}")
        print(f"  Major: {delta['major']}, Minor: {delta['minor']}, Patch: {delta['patch']}")

        if delta.get('error'):
            print(f"  ⚠️ Error: {delta['error']}")
            return

        # Map semantic meaning to visual indicators
        state_mapping: Dict[str, Tuple[str, str]] = {
            "upgrade": ("🔼", "Upgrade"),
            "downgrade": ("🔽", "Downgrade"),
            "equivalent": ("⏸️", "Equivalent")
        }

        state_key = (
            "upgrade" if delta.get('is_upgrade') else
            ("downgrade" if delta.get('is_downgrade') else "equivalent")
        )

        emoji, label = state_mapping[state_key]
        print(f"  {emoji} {label}")

    # ┌────────────────────────────────────────────────────────┐
    # │ Command Implementations - Pure functions with effects  │
    # └────────────────────────────────────────────────────────┘
    def cmd_get(_: List[str]) -> None:
        """Display current version in canonical format"""
        print(f"Version: {format_version(version_config.__version__)}")

    def cmd_check(args: List[str]) -> None:
        """Verify compatibility between versions"""
        if not args:
            print("❌ Missing version argument")
            return

        version = args[0]
        is_valid = is_compatible(version)
        status = "✓ compatible" if is_valid else "✗ incompatible"
        print(f"Version {version} is {status}")

    def cmd_update(args: List[str]) -> None:
        """Apply version changes universally"""
        if not args:
            print("❌ Missing version argument")
            return

        result = update_version(args[0])

        if result["updated"]:
            print(f"✅ Updated {len(result['files_changed'])} files to {result['version_to']}")
            print(f"   Duration: {result['duration_seconds']:.2f}s")
        else:
            print(f"ℹ️ No changes needed for version {result['version_to']}")

    def cmd_status(_: List[str]) -> None:
        """Display comprehensive version diagnostics"""
        print(json.dumps(get_version_status(), indent=2))

    def cmd_compare(args: List[str]) -> None:
        """Compare two versions with visual indicators"""
        if len(args) < 2:
            print("❌ Need two versions to compare (v1 v2)")
            return

        print_comparison(args[0], args[1])

    def cmd_help(_: List[str]) -> None:
        """Display available commands with usage instructions"""
        print(f"📦 Current: {format_version(version_config.__version__)}")

        print("\n┌─────────────────────────────────────────────────┐")
        print("│ Available Commands                            │")
        print("└─────────────────────────────────────────────────┘")

        commands_help = [
            ("get", "Display current version"),
            ("check <VERSION>", "Verify version compatibility"),
            ("update <VERSION>", "Apply version universally"),
            ("compare <V1> <V2>", "Analyze version differences"),
            ("status", "Show diagnostic information"),
            ("help", "Display this help message")
        ]

        # Display commands with consistent formatting
        max_cmd_width = max(len(cmd) for cmd, _ in commands_help)
        for cmd, desc in commands_help:
            print(f"  {cmd.ljust(max_cmd_width + 2)} - {desc}")

    # ┌────────────────────────────────────────────────────────┐
    # │ Command Registration - Type-safe dispatch table        │
    # └────────────────────────────────────────────────────────┘
    commands: Dict[str, CommandHandler] = {
        "get": cmd_get,
        "check": cmd_check,
        "update": cmd_update,
        "status": cmd_status,
        "compare": cmd_compare,
        "help": cmd_help
    }

    # ┌────────────────────────────────────────────────────────┐
    # │ Command Execution - Parse and dispatch                 │
    # └────────────────────────────────────────────────────────┘
    display_header()

    if len(sys.argv) > 1 and (cmd := sys.argv[1]) in commands:
        commands[cmd](sys.argv[2:])
    else:
        # Default to help when command is missing or invalid
        cmd_help([])
