"""
Command-line interface for version_forge operations with universal Eidosian patterns.

Provides elegant semantic interactions for version management across all Eidosian forge projects
with recursive validation, dependency checking, and type-safe operations.
"""
import argparse
import json
import logging
import sys
import time
from functools import wraps
from pathlib import Path
from typing import (Any, Callable, Dict, Final, List, Protocol, Tuple, TypeVar,
                    cast)

from ..compatibility.validator import DependencyValidator
from ..core.config import VersionConfig
from ..core.version import SimpleVersion, format_version
from ..operations.compare import (calculate_delta, calculate_version_delta,
                                  is_compatible)
from ..operations.migration import MigrationGuideGenerator
from ..operations.update import CompleteVersionUpdateResult, update_version
from ..protocols.interfaces import VersionDelta

# Type definitions for enhanced precision
T = TypeVar('T')
CommandResult = Tuple[bool, str]
CommandHandler = Callable[[argparse.Namespace], int]

# Constants with clear semantic intent - emojis as signposts, not decorations
EMOJI_SUCCESS: Final[str] = "✅"
EMOJI_INFO: Final[str] = "ℹ️"
EMOJI_WARNING: Final[str] = "⚠️"
EMOJI_ERROR: Final[str] = "❌"
EMOJI_CHART: Final[str] = "📊"
EMOJI_UP: Final[str] = "🔼"
EMOJI_DOWN: Final[str] = "🔽"
EMOJI_SAME: Final[str] = "⏸️"

logger: logging.Logger = logging.getLogger("forge.version")

# Function decorator for timing and logging execution
def time_execution(func: Callable[..., T]) -> Callable[..., T]:
    """Measure execution time for performance analysis."""
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        start = time.time()
        try:
            return func(*args, **kwargs)
        finally:
            logger.debug("%s executed in %.3fs", func.__name__, time.time() - start)
    return wrapper

def setup_logging(debug: bool = False) -> None:
    """Configure logging with appropriate verbosity and type-aware formatting."""
    level: int = logging.DEBUG if debug else logging.INFO
    format_string: str = (
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s" if debug
        else "%(levelname)s: %(message)s"
    )

    logging.basicConfig(level=level, format=format_string)
    logger.debug("Logging initialized with debug mode: %s", "enabled" if debug else "disabled")

def get_version_command(args: argparse.Namespace) -> int:
    """
    Display current version information with fractal completeness.

    Shows structured version data with expanding detail based on verbosity.
    """
    config = VersionConfig()
    formatted: str = format_version(config.__version__)

    # Basic version display with elegant framing
    print(f"{formatted} {'(' + config.source + ')' if args.verbose else ''}")

    if args.verbose:
        # Display structured information with visual boundaries for clarity
        fields = [
            ("Major", config.major),
            ("Minor", config.minor),
            ("Patch", config.patch),
            ("Min version", config.min_version),
            ("Release", config.release_date),
            ("Source", config.source)
        ]

        width = max(len(name) + len(str(value)) for name, value in fields) + 10
        print(f"\n┌{'─' * width}┐")
        for name, value in fields:
            print(f"│ {name}:{' ' * (width - len(name) - len(str(value)) - 4)}{value} │")
        print(f"└{'─' * width}┘")

    return 0

def check_version_command(args: argparse.Namespace) -> int:
    """
    Check version compatibility with structural elegance.

    Verifies if a version satisfies compatibility constraints
    and provides visual feedback with emoji indicators.
    """
    if not args.version:
        print(f"{EMOJI_ERROR} Version argument is required")
        return 1

    config = VersionConfig()
    is_valid: bool = is_compatible(args.version, config.min_version)
    delta: VersionDelta = calculate_delta(args.version, config.min_version)

    if is_valid:
        print(f"{EMOJI_SUCCESS} Version {args.version} is compatible with minimum {config.min_version}")

        # Show upgrade margin when compatible
        if not delta.get("is_same", False):
            margin = []
            if delta.get("major", 0) > 0:
                margin.append(f"{delta.get('major')} major")
            if delta.get("minor", 0) > 0:
                margin.append(f"{delta.get('minor')} minor")
            if delta.get("patch", 0) > 0:
                margin.append(f"{delta.get('patch')} patch")

            print(f"  {EMOJI_UP} Version is ahead by {', '.join(margin)}")
    else:
        print(f"{EMOJI_ERROR} Version {args.version} is incompatible with minimum {config.min_version}")

        # Provide helpful context on version delta for debugging
        if delta.get("is_downgrade", False):
            behind_parts = []
            if abs(delta.get("major", 0)) > 0:
                behind_parts.append(f"{abs(delta.get('major', 0))} major")
            if abs(delta.get("minor", 0)) > 0:
                behind_parts.append(f"{abs(delta.get('minor', 0))} minor")
            if abs(delta.get("patch", 0)) > 0:
                behind_parts.append(f"{abs(delta.get('patch', 0))} patch")

            print(f"  {EMOJI_DOWN} Version is behind by {', '.join(behind_parts)}")
            print(f"  Suggestion: Update to at least {config.min_version}")

    return 0 if is_valid else 1

@time_execution
def update_version_command(args: argparse.Namespace) -> int:
    """
    Update version throughout the codebase with surgical precision.

    Performs intelligent pattern recognition to update version references
    while preserving structural integrity across files.
    """
    if not args.version:
        print(f"{EMOJI_ERROR} Version argument is required")
        return 1

    config = VersionConfig()
    current_version: str = config.__version__

    try:
        # Validate version format with improved type safety
        version_obj = SimpleVersion(args.version)

        # Version validation in one concise check
        if not all(hasattr(version_obj, attr) for attr in ('major', 'minor', 'patch')):
            print(f"{EMOJI_ERROR} Invalid version format: {args.version}")
            return 1

        # Execute update operation
        result: CompleteVersionUpdateResult = update_version(
            args.version,
            current_version,
            Path(args.repo) if args.repo else None,
        )

        if result["updated"]:
            file_count: int = len(result["files_changed"])
            print(f"{EMOJI_SUCCESS} Updated {file_count} file{'s' if file_count != 1 else ''} "
                  f"from {result['previous_version']} → {result['current_version']}")
            print(f"   Duration: {result['duration_seconds']:.2f}s")

            if args.verbose and result["files_changed"]:
                print("\nFiles changed:")
                for file in sorted(result["files_changed"]):
                    print(f"  - {file}")

            if args.verbose:
                # Provide detailed statistics
                examined: int = max(1, result["files_examined"])  # Prevent division by zero
                efficiency: float = len(result["files_changed"]) / examined * 100
                print(f"\nScanning efficiency: {len(result['files_changed'])}/{result['files_examined']} "
                      f"({efficiency:.1f}%)")

                # Add witty efficiency comment
                if efficiency < 10:
                    print("  Looking for versions like finding needles in a haystack factory.")
                elif efficiency > 50:
                    print("  Impressive hit rate! Your versioning is well-organized.")
        else:
            print(f"{EMOJI_INFO} No changes needed for version {result['current_version']}")

        return 0
    except Exception as e:
        print(f"{EMOJI_ERROR} Error updating version: {e}")
        logger.exception("Version update failed")
        return 1

def compare_versions_command(args: argparse.Namespace) -> int:
    """
    Compare two versions with visual precision and semantic understanding.

    Analyzes version differences with structured output and intelligent
    categorization of changes based on semantic versioning principles.
    """
    if not args.version1 or not args.version2:
        print(f"{EMOJI_ERROR} Both version arguments are required")
        return 1

    try:
        # Use the semantic version delta calculator
        delta: VersionDelta = calculate_version_delta(args.version1, args.version2)

        # Output with visual structure
        print(f"{EMOJI_CHART} {args.version1} → {args.version2}")

        # Delta components with sign-awareness for clearer representation
        parts = []
        for component in ("major", "minor", "patch"):
            value = delta.get(component, 0)
            if value != 0:
                parts.append(f"{component.capitalize()}: {'+' if value > 0 else ''}{value}")

        print("  " + (", ".join(parts) if parts else "No version difference"))

        # Semantic categorization with visual indicators
        if delta.get("is_upgrade", False):
            change_type = "Major" if delta.get("major", 0) > 0 else "Minor" if delta.get("minor", 0) > 0 else "Patch"
            impact = {
                "Major": "expect breaking changes",
                "Minor": "new features",
                "Patch": "bug fixes"
            }
            print(f"  {EMOJI_UP} Upgrade ({change_type} - {impact[change_type]})")

        elif delta.get("is_downgrade", False):
            change_type = "Major" if delta.get("major", 0) < 0 else "Minor" if delta.get("minor", 0) < 0 else "Patch"
            impact = {
                "Major": "significant rollback",
                "Minor": "feature removal",
                "Patch": "reverting fixes"
            }
            print(f"  {EMOJI_DOWN} Downgrade ({change_type} - {impact[change_type]})")

        elif delta.get("is_same", False):
            print(f"  {EMOJI_SAME} Equivalent versions")

        if args.verbose:
            # Migration path suggestion for smart upgrades
            print("\nSuggested migration path:")
            if delta.get("major", 0) > 0:
                print("  • Review breaking changes in documentation")
                print("  • Update dependencies before upgrading")
                print("  • Consider incremental updates through minor versions")
            elif delta.get("minor", 0) > 0:
                if delta.get("minor", 0) > 3:
                    print("  • Test new features incrementally")
                    print("  • Review deprecation notices")
                else:
                    print("  • Update with standard testing procedures")
            elif delta.get("is_downgrade", False) and delta.get("major", 0) < 0:
                print("  • Caution: Major downgrade may result in lost functionality")
                print("  • Create compatibility layer for dependent systems")

        return 0

    except Exception as e:
        print(f"{EMOJI_ERROR} Error comparing versions: {e}")
        logger.exception("Version comparison failed")
        return 1

class VersionLike(Protocol):
    """Protocol for objects with semantic version components."""
    major: int
    minor: int
    patch: int

# Component version wrapper - moved outside function for cleaner organization
class VersionWrapper:
    """Wrapper for component version info implementing IVersioned protocol.

    Adapts raw version data into a properly typed version object with
    comparable interface and standardized access patterns.
    """

    def __init__(self, ver_data: Dict[str, Any]) -> None:
        """Initialize with component version data dictionary.

        Args:
            ver_data: Component metadata including version information
        """
        version_str = ver_data.get("version", "0.1.0")
        # Parse string into proper version object
        self._version_object = SimpleVersion(version_str)
        self.version_info = ver_data

    @property
    def version(self) -> SimpleVersion:
        """Version object implementing required version protocol."""
        return self._version_object

    @property
    def major(self) -> int:
        """Major version component."""
        return self._version_object.major

    @property
    def minor(self) -> int:
        """Minor version component."""
        return self._version_object.minor

    @property
    def patch(self) -> int:
        """Patch version component."""
        return self._version_object.patch

    def __eq__(self, other: object) -> bool:
        """Compare versions for equality."""
        if not hasattr(other, 'major') or not hasattr(other, 'minor') or not hasattr(other, 'patch'):
            return NotImplemented

        versioned_other = cast(VersionLike, other)
        return (self.major == versioned_other.major and
                self.minor == versioned_other.minor and
                self.patch == versioned_other.patch)
    def __lt__(self, other: object) -> bool:
        """Compare versions for ordering."""
        if not hasattr(other, 'major') or not hasattr(other, 'minor') or not hasattr(other, 'patch'):
            return NotImplemented

        # Compare major first, then minor, then patch
        versioned_other = cast(VersionLike, other)
        if self.major != versioned_other.major:
            return self.major < versioned_other.major
        if self.minor != versioned_other.minor:
            return self.minor < versioned_other.minor
        return self.patch < versioned_other.patch

        # Compare major first, then minor, then patch
        if self.major != other.major:
            return self.major < other.major
        if self.minor != other.minor:
            return self.minor < other.minor
        return self.patch < other.patch

def _load_components_from_file(validator: DependencyValidator, file_path: Path) -> CommandResult:
    """Load component definitions from JSON file."""
    try:
        with open(file_path) as f:
            component_data = json.load(f)

        # Register components with proper typing
        for name, data in component_data.get("components", {}).items():
            validator.register_component(name, VersionWrapper(data))

        # Register dependencies
        for dep in component_data.get("dependencies", []):
            if "from" in dep and "to" in dep:
                validator.register_dependency(dep["from"], dep["to"])
                logger.debug("Registered dependency: %s → %s", dep["from"], dep["to"])

        return True, f"Loaded {len(component_data.get('components', {}))} components and {len(component_data.get('dependencies', []))} dependencies"
    except (json.JSONDecodeError, KeyError) as e:
        return False, f"Invalid components file format: {e}"
    except Exception as e:
        return False, f"Failed to load components: {e}"

def _scan_directory_for_components(validator: DependencyValidator, scan_dir: Path) -> CommandResult:
    """Scan directory for components and dependencies."""
    components_found = 0
    dependencies_found = 0

    # Look for version markers in files
    version_patterns = list(scan_dir.glob("**/version*.py"))
    init_patterns = list(scan_dir.glob("**/__init__.py"))
    package_patterns = list(scan_dir.glob("**/package.json"))

    # Extract version info from Python files
    for file_path in version_patterns + init_patterns:
        try:
            # Simple regex-based version extraction would be implemented here
            # For now, we just note the potential component
            logger.debug("Found potential component at: %s", file_path.parent)
            components_found += 1
        except Exception as e:
            logger.warning("Failed to process %s: %s", file_path, e)

    # Extract from package.json files
    for file_path in package_patterns:
        try:
            with open(file_path) as f:
                pkg_data = json.load(f)
                if "dependencies" in pkg_data:
                    dependencies_found += len(pkg_data["dependencies"])
                    # Would register dependencies here in a complete implementation
        except Exception as e:
            logger.warning("Failed to process %s: %s", file_path, e)

    if components_found == 0:
        return False, "No components found in directory scan"

    return True, f"Found {components_found} potential components and {dependencies_found} dependencies"

def validate_command(args: argparse.Namespace) -> int:
    """
    Validate dependency relationships across components with recursive precision.

    Performs topological analysis of dependency graphs to ensure version
    compatibility across the Eidosian ecosystem with elegant error reporting.
    """
    validator = DependencyValidator()

    # Handle component loading from different sources
    if args.components_file:
        # Load components from file
        file_path = Path(args.components_file)
        if not file_path.exists():
            print(f"{EMOJI_ERROR} Components file not found: {file_path}")
            return 1

        success, message = _load_components_from_file(validator, file_path)
        if not success:
            print(f"{EMOJI_ERROR} {message}")
            return 1
        print(f"{EMOJI_INFO} {message}")

    elif args.scan_directory:
        scan_dir = Path(args.scan_directory)
        if not scan_dir.is_dir():
            print(f"{EMOJI_ERROR} Invalid scan directory: {scan_dir}")
            return 1

        print(f"{EMOJI_INFO} Scanning {scan_dir} for components...")
        success, message = _scan_directory_for_components(validator, scan_dir)
        print(f"{EMOJI_INFO if success else EMOJI_WARNING} {message}")

        if not success:
            print(f"{EMOJI_WARNING} Directory scanning yielded insufficient data - consider using a components file")
    else:
        print(f"{EMOJI_ERROR} Must specify either --components-file or --scan-directory")
        return 1

    # Perform validation with comprehensive error collection
    is_valid, errors = validator.validate_dependency_graph()

    if is_valid:
        print(f"{EMOJI_SUCCESS} Dependency graph is valid")
        return 0
    else:
        print(f"{EMOJI_ERROR} Dependency graph validation failed:")
        for error in errors:
            print(f"  • {error}")

        if args.fix and args.target_versions:
            # Try to generate upgrade plan
            try:
                target_versions = json.loads(args.target_versions)
                plan = validator.get_upgrade_plan(target_versions)

                if plan:
                    print("\n📋 Suggested upgrade plan:")
                    for component, version in plan.items():
                        print(f"  • {component}: → {version}")
                else:
                    print(f"{EMOJI_WARNING} No viable upgrade plan found")
            except json.JSONDecodeError:
                print(f"{EMOJI_ERROR} Invalid target versions format. Expected JSON object.")

        return 1

@time_execution
def migration_guide_command(args: argparse.Namespace) -> int:
    """
    Generate migration guide between versions with semantic intelligence.

    Creates structured, actionable migration paths with estimated effort levels
    and automatic suggestions based on version deltas.
    """
    if not args.component or not args.from_version or not args.to_version:
        print(f"{EMOJI_ERROR} Component name, from-version and to-version are required")
        return 1

    try:
        # Initialize migration guide generator
        generator = MigrationGuideGenerator()

        # Generate migration guide
        guide = generator.generate_migration_guide(
            args.component,
            args.from_version,
            args.to_version
        )

        # Output with visual structure - aligned for readability
        print(f"🧭 Migration Guide: {args.component}")
        print(f"  From: {guide['from_version']} → To: {guide['to_version']}")
        print(f"  Type: {guide['upgrade_type'].upper()}")
        print(f"  Effort: {guide['estimated_effort'].upper()}")

        # Display detailed sections with elegant framing
        def print_section(title: str, items: List[str]) -> None:
            """Print a section of the migration guide with consistent formatting."""
            if not items:
                return

            print(f"\n{title}:")
            for item in items:
                print(f"  • {item}")

        # Order sections by importance
        print_section("🚨 Breaking Changes", guide["breaking_changes"])
        print_section("✨ New Features", guide["new_features"])
        print_section("⚠️ Deprecations", guide["deprecations"])
        print_section("💡 Suggestions", guide["suggestions"])

        return 0

    except Exception as e:
        print(f"{EMOJI_ERROR} Error generating migration guide: {e}")
        logger.exception("Migration guide generation failed")
        return 1

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments with elegant structure and clear intentions.

    Creates a seamless command interface with intelligent grouping and
    contextually relevant parameter validation.
    """
    parser = argparse.ArgumentParser(
        description="Eidosian version management and compatibility toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
┌──────────────────────────────────────────────┐
│ Examples:                                    │
│   version_forge get -v                       │
│   version_forge update 1.2.3                 │
│   version_forge compare 1.0.0 2.0.0          │
│   version_forge validate --components-file... │
└──────────────────────────────────────────────┘
        """
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Get version command
    get_parser = subparsers.add_parser("get", help="Display current version")
    get_parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed information")

    # Check version command
    check_parser = subparsers.add_parser("check", help="Check version compatibility")
    check_parser.add_argument("version", help="Version to check against minimum")

    # Update version command
    update_parser = subparsers.add_parser("update", help="Update version references")
    update_parser.add_argument("version", help="New version to set")
    update_parser.add_argument("--repo", help="Repository path to update")
    update_parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed output")

    # Compare versions command
    compare_parser = subparsers.add_parser("compare", help="Compare two versions")
    compare_parser.add_argument("version1", help="Base version")
    compare_parser.add_argument("version2", help="Target version")
    compare_parser.add_argument("-v", "--verbose", action="store_true", help="Show migration suggestions")

    # Validate dependencies command
    validate_parser = subparsers.add_parser("validate", help="Validate dependency graph")
    validate_source = validate_parser.add_mutually_exclusive_group(required=True)
    validate_source.add_argument("--components-file", help="JSON file with component definitions")
    validate_source.add_argument("--scan-directory", help="Directory to scan for components")
    validate_parser.add_argument("--fix", action="store_true", help="Suggest fixes for validation issues")
    validate_parser.add_argument("--target-versions", help="JSON string with target versions to upgrade to")

    # Migration guide command
    migrate_parser = subparsers.add_parser("migrate", help="Generate migration guide")
    migrate_parser.add_argument("component", help="Component name")
    migrate_parser.add_argument("from_version", help="Source version")
    migrate_parser.add_argument("to_version", help="Target version")

    return parser.parse_args()

def main() -> int:
    """
    Main entry point for CLI with elegant error handling and command routing.

    Provides a unified interface to version management operations with
    contextually aware command handling and precise execution.
    """
    try:
        args = parse_args()
        setup_logging(args.debug)

        # Command handler mapping with precise typing
        command_handlers: Dict[str, CommandHandler] = {
            "get": get_version_command,
            "check": check_version_command,
            "update": update_version_command,
            "compare": compare_versions_command,
            "validate": validate_command,
            "migrate": migration_guide_command
        }

        if args.command in command_handlers:
            logger.debug("Executing command: %s", args.command)
            return command_handlers[args.command](args)
        else:
            print(f"{EMOJI_ERROR} No command specified. Use --help for usage information.")
            return 1
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 130  # Standard exit code for Ctrl+C
    except Exception as e:
        logger.exception("Unhandled exception in main")
        print(f"{EMOJI_ERROR} Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
