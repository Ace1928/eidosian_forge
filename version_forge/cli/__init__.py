"""
Command-line interface for version_forge operations with universal Eidosian patterns.

Provides elegant semantic interactions for version management across all Eidosian forge projects
with recursive validation, dependency checking, and type-safe operations.
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Callable, Dict, Final, List, Literal, LiteralString

from ..compatibility.validator import DependencyValidator
from ..core.config import VersionConfig
from ..core.version import SimpleVersion, format_version
from ..operations.compare import (calculate_delta, calculate_version_delta,
                                  is_compatible)
from ..operations.migration import MigrationGuideGenerator
from ..operations.update import CompleteVersionUpdateResult, update_version
from ..protocols.interfaces import VersionDelta

# Type aliases for command handlers for precision
CommandHandler = Callable[[argparse.Namespace], int]

# Constants with clear semantic intent
EMOJI_SUCCESS: Final[str] = "✅"
EMOJI_INFO: Final[str] = "ℹ️"
EMOJI_WARNING: Final[str] = "⚠️"
EMOJI_ERROR: Final[str] = "❌"
EMOJI_CHART: Final[str] = "📊"
EMOJI_UP: Final[str] = "🔼"
EMOJI_DOWN: Final[str] = "🔽"
EMOJI_SAME: Final[str] = "⏸️"

logger: logging.Logger = logging.getLogger("forge.version")

def setup_logging(debug: bool = False) -> None:
    """Configure logging with appropriate verbosity and type-aware formatting."""
    level: Literal[10] | Literal[20] = logging.DEBUG if debug else logging.INFO
    format_string: Literal['%(levelname)s: %(message)s'] | LiteralString = "%(levelname)s: %(message)s" if not debug else \
                    "%(asctime)s | %(name)s | %(levelname)s | %(message)s"

    logging.basicConfig(
        level=level,
        format=format_string
    )

    logger.debug("Initialized logging with debug mode: %s", "enabled" if debug else "disabled")

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
        # Structured formatting for clarity
        print(f"\n┌{'─' * 30}┐")
        print(f"│ Major:       {config.major:<18} │")
        print(f"│ Minor:       {config.minor:<18} │")
        print(f"│ Patch:       {config.patch:<18} │")
        print(f"│ Min version: {config.min_version:<18} │")
        print(f"│ Release:     {config.release_date:<18} │")
        print(f"│ Source:      {config.source:<18} │")
        print(f"└{'─' * 30}┘")

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

    if is_valid:
        print(f"{EMOJI_SUCCESS} Version {args.version} is compatible with minimum {config.min_version}")
    else:
        print(f"{EMOJI_ERROR} Version {args.version} is incompatible with minimum {config.min_version}")

        # Provide helpful context on version delta for debugging
        delta: VersionDelta = calculate_delta(args.version, config.min_version)
        if delta.get("is_downgrade", False):
            print(f"  {EMOJI_DOWN} Version is {abs(delta.get('major', 0))} major, {abs(delta.get('minor', 0))} minor, "
                  f"and {abs(delta.get('patch', 0))} patch versions behind minimum")

    return 0 if is_valid else 1

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

        # Remove duplicate initialization and use correct validation
        # Option 1: Check if version has valid structure
        if not hasattr(version_obj, 'major') or not hasattr(version_obj, 'minor') or not hasattr(version_obj, 'patch'):
            print(f"{EMOJI_ERROR} Invalid version format: {args.version}")
            return 1

        # Option 2: Try direct validation (if SimpleVersion has _is_valid internal property)
        # if not getattr(version_obj, '_valid', False):
        #    print(f"{EMOJI_ERROR} Invalid version format: {args.version}")
        #    return 1

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

        # Delta components with sign-awareness
        major_change: str = f"{'+' if delta.get('major', 0) > 0 else ''}{delta.get('major', 0)}"
        minor_change: str = f"{'+' if delta.get('minor', 0) > 0 else ''}{delta.get('minor', 0)}"
        patch_change: str = f"{'+' if delta.get('patch', 0) > 0 else ''}{delta.get('patch', 0)}"

        print(f"  Major: {major_change}, Minor: {minor_change}, Patch: {patch_change}")

        # Semantic categorization with visual indicators
        if delta.get("is_upgrade", False):
            print(f"  {EMOJI_UP} Upgrade " +
                  ("(Major - expect breaking changes)" if delta.get("major", 0) > 0 else
                   "(Minor - new features)" if delta.get("minor", 0) > 0 else
                   "(Patch - bug fixes)"))
        elif delta.get("is_downgrade", False):
            print(f"  {EMOJI_DOWN} Downgrade " +
                  ("(Major - significant rollback)" if delta.get("major", 0) < 0 else
                   "(Minor - feature removal)" if delta.get("minor", 0) < 0 else
                   "(Patch - reverting fixes)"))
        elif delta.get("is_same", False):
            print(f"  {EMOJI_SAME} Equivalent versions")

        if args.verbose:
            # Migration path suggestion for smart upgrades
            print("\nSuggested migration path:")
            if delta.get("major", 0) > 0:
                print("  • Review breaking changes in documentation")
                print("  • Update dependencies before upgrading")
                print("  • Consider incremental updates through minor versions")
            elif delta.get("minor", 0) > 0 and delta.get("minor", 0) > 3:
                print("  • Test new features incrementally")
                print("  • Review deprecation notices")

        return 0

    except Exception as e:
        print(f"{EMOJI_ERROR} Error comparing versions: {e}")
        logger.exception("Version comparison failed")
        return 1

def validate_command(args: argparse.Namespace) -> int:
    """
    Validate dependency relationships across components with recursive precision.

    Performs topological analysis of dependency graphs to ensure version
    compatibility across the Eidosian ecosystem with elegant error reporting.
    """
    try:
        validator = DependencyValidator()

        # Handle component loading from different sources
        if args.components_file:
            # Load components from file
            file_path = Path(args.components_file)
            if not file_path.exists():
                print(f"{EMOJI_ERROR} Components file not found: {file_path}")
                return 1

            try:
                with open(file_path) as f:
                    component_data = json.load(f)

                # Register components and dependencies
                for name, data in component_data.get("components", {}).items():
                    # Create component wrapper that implements IVersioned protocol
                    from typing import Any, Dict

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
                        def version(self) -> SimpleVersion:  # Return the actual version object, not a string
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

                        def __str__(self) -> str:
                            """String representation with version prefix."""
                            return f"v{str(self._version_object)}"

                        def __eq__(self, other: object) -> bool:
                            """Compare versions for equality.

                            Performs structural comparison against any object with compatible attributes.
                            """
                            if hasattr(other, 'major') and hasattr(other, 'minor') and hasattr(other, 'patch'):
                                other_major = getattr(other, 'major')
                                other_minor = getattr(other, 'minor')
                                other_patch = getattr(other, 'patch')
                                if all(isinstance(x, int) for x in (other_major, other_minor, other_patch)):
                                    return (self.major == other_major and
                                            self.minor == other_minor and
                                            self.patch == other_patch)
                            return NotImplemented

                        def __lt__(self, other: object) -> bool:
                            """Compare versions for ordering.

                            Implements semantic versioning comparison logic for sorting.
                            """
                            if hasattr(other, 'major') and hasattr(other, 'minor') and hasattr(other, 'patch'):
                                other_major = getattr(other, 'major')
                                other_minor = getattr(other, 'minor')
                                other_patch = getattr(other, 'patch')
                                if all(isinstance(x, int) for x in (other_major, other_minor, other_patch)):
                                    # Compare major first, then minor, then patch
                                    if self.major != other_major:
                                        return self.major < other_major
                                    if self.minor != other_minor:
                                        return self.minor < other_minor
                                    return self.patch < other_patch
                            return NotImplemented

                    validator.register_component(name, VersionWrapper(data))

                # Register dependencies with validating all relationships
                for dep in component_data.get("dependencies", []):
                    if "from" in dep and "to" in dep:
                        validator.register_dependency(dep["from"], dep["to"])
                        logger.debug(f"Registered dependency: {dep['from']} → {dep['to']}")

            except (json.JSONDecodeError, KeyError) as e:
                print(f"{EMOJI_ERROR} Invalid components file format: {e}")
                return 1
        elif args.scan_directory:
            scan_dir = Path(args.scan_directory)
            if not scan_dir.is_dir():
                print(f"{EMOJI_ERROR} Invalid scan directory: {scan_dir}")
                return 1

            print(f"{EMOJI_INFO} Scanning {scan_dir} for components...")

            # Find components in directories through version markers
            components_found = 0
            dependencies_found = 0

            # Scan for version.py, __init__.py, package.json, etc.
            for file_path in scan_dir.glob("**/version*.py") or scan_dir.glob("**/__init__.py"):
                # Extract version info from file - simplified approach
                print(f"  Found potential component at: {file_path.parent}")
                components_found += 1

            if components_found == 0:
                print(f"{EMOJI_WARNING} No components found in directory scan")
            else:
                print(f"{EMOJI_INFO} Found {components_found} potential components")
                print(f"{EMOJI_INFO} Detected {dependencies_found} dependencies")

            # Note: Full directory scanning would require more complex implementation
            print(f"{EMOJI_WARNING} Directory scanning is limited - use components file for complete validation")
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
                except json.JSONDecodeError:
                    print(f"{EMOJI_ERROR} Invalid target versions format. Expected JSON object.")

            return 1

    except Exception as e:
        print(f"{EMOJI_ERROR} Validation error: {e}")
        logger.exception("Dependency validation failed")
        return 1

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

        # Output with visual structure
        print(f"🧭 Migration Guide: {args.component}")
        print(f"  From: {guide['from_version']} → To: {guide['to_version']}")
        print(f"  Type: {guide['upgrade_type'].upper()}")
        print(f"  Effort: {guide['estimated_effort'].upper()}")

        # Display detailed sections with elegant framing
        def print_section(title: str, items: List[str]) -> None:
            if not items:
                return

            print(f"\n{title}:")
            for item in items:
                print(f"  • {item}")

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
            logger.debug(f"Executing command: {args.command}")
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
