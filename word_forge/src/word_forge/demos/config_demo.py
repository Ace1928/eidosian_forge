"""
Demonstration and CLI tool for Word Forge configuration.
"""

import argparse
import json

from word_forge.config import config  # Import necessary items

# Import serialize_config explicitly
from word_forge.configs.config_essentials import ConfigValue, serialize_config


def main() -> None:
    """
    Display current configuration settings.

    Command-line interface function that provides options to validate
    configuration, export to file, display configuration components,
    and perform advanced configuration operations.

    Usage:
        python -m word_forge.demos.config_demo --validate
        python -m word_forge.demos.config_demo --export config.json
        python -m word_forge.demos.config_demo --component database
        python -m word_forge.demos.config_demo --sources  # Show where settings came from
        python -m word_forge.demos.config_demo --profile production  # Apply a profile
    """
    parser = argparse.ArgumentParser(description="Word Forge Configuration Utility")

    # Basic options
    parser.add_argument(
        "--component",
        "-c",
        help="Display specific component configuration",
        choices=config.get_available_components(),
    )
    parser.add_argument(
        "--validate",
        "-v",
        action="store_true",
        help="Validate configuration",
    )
    parser.add_argument(
        "--export",
        "-e",
        help="Export configuration to JSON file",
    )

    # Enhanced options
    parser.add_argument(
        "--sources",
        "-s",
        action="store_true",
        help="Show configuration value sources",
    )
    parser.add_argument(
        "--profile",
        "-p",
        help="Apply a configuration profile",
        choices=[
            "development",
            "production",
            "testing",
            "high_performance",
            "low_memory",
        ],
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show status of all components",
    )
    parser.add_argument(
        "--set",
        "-S",
        nargs=3,
        metavar=("COMPONENT", "SETTING", "VALUE"),
        help="Set a configuration value (e.g. database db_path /path/to/db.sqlite)",
    )

    args = parser.parse_args()

    # Apply profile if specified (do this first as it may affect other operations)
    if args.profile:
        try:
            config.apply_profile(args.profile)
            print(f"✅ Applied configuration profile: {args.profile}")
        except Exception as e:
            print(f"❌ Failed to apply profile: {str(e)}")
            return

    # Set a specific value if requested
    if args.set:
        component_name, attr_name, value_str = args.set
        try:
            # Get the current value to determine type
            component = config.get_component(component_name)
            if not component:
                print(f"❌ Component '{component_name}' not found")
                return

            if not hasattr(component, attr_name):
                print(
                    f"❌ Attribute '{attr_name}' not found in component '{component_name}'"
                )
                return

            current_value = getattr(component, attr_name)

            # Convert the string value to the appropriate type
            # Need to import Enum for this check
            from enum import Enum

            if isinstance(current_value, bool):
                value = value_str.lower() in ("true", "yes", "1", "y")
            elif isinstance(current_value, int):
                value = int(value_str)
            elif isinstance(current_value, float):
                value = float(value_str)
            elif isinstance(current_value, Enum):
                # Assuming the Enum class is accessible via current_value.__class__
                value = current_value.__class__(value_str)
            else:
                value = value_str

            # Set the value
            config.set_runtime_value(component_name, attr_name, value)
            print(f"✅ Set {component_name}.{attr_name} = {value!r}")
        except Exception as e:
            print(f"❌ Failed to set value: {str(e)}")
            return

    # Validate if requested
    if args.validate:
        validation_results = config.validate_all()
        invalid_components = {
            comp: errors for comp, errors in validation_results.items() if errors
        }

        if invalid_components:
            print("❌ Configuration validation failed:")
            for component, errors in invalid_components.items():
                print(f"  • {component}: {'; '.join(errors)}")
        else:
            print("✅ Configuration validation passed for all components")

    # Export if requested
    if args.export:
        try:
            config.export_to_file(args.export)
            print(f"✅ Configuration exported to {args.export}")
        except Exception as e:
            print(f"❌ Export failed: {str(e)}")
            return

    # Show component status if requested
    if args.status:
        print("Component Status:")
        print("----------------")

        for component_name in sorted(config.get_available_components()):
            status = config.get_component_status(component_name)
            validation = status["validation"]
            accessed = "✓" if status["accessed"] else "✗"
            errors = status["error_count"]

            status_icon = (
                "✅"
                if validation == "valid"
                else "⚠️" if validation.startswith("not") else "❌"
            )

            print(
                f"{status_icon} {component_name}: Accessed={accessed}, Errors={errors}"
            )

            # Show dependencies if any
            if status["dependencies"]:
                deps = ", ".join(status["dependencies"])
                print(f"   Dependencies: {deps}")
        print()

    # Show sources if requested
    if args.sources:
        print("Configuration Value Sources:")
        print("---------------------------")

        for component_name in sorted(config.get_available_components()):
            component = config.get_component(component_name)
            if not component:
                continue
            print(f"\n{component_name.capitalize()} Component:")

            # Get all non-private attributes
            for attr_name in dir(component):
                if attr_name.startswith("_") or callable(getattr(component, attr_name)):
                    continue

                try:
                    value, source = config.get_value_with_source(
                        component_name, attr_name
                    )
                    source_type = source.type.name
                    location = f" ({source.location})" if source.location else ""

                    # Format value for display
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        value_str = repr(value)
                    else:
                        value_str = f"{type(value).__name__} instance"

                    print(f"  {attr_name}: {value_str}")
                    print(f"    Source: {source_type}{location}")
                except Exception:
                    # Skip attributes that can't be accessed
                    continue
        print()

    # Show specific component if requested
    if args.component:
        component = config.get_component(args.component)
        if component:
            print(f"{args.component.title()} Configuration")
            print("=" * (len(args.component) + 14))

            # Serialize and print component config
            component_dict: ConfigValue = serialize_config(
                component
            )  # Use ConfigValue type hint
            # Ensure component_dict is a dictionary before passing to json.dumps
            if isinstance(component_dict, dict):
                print(json.dumps(component_dict, indent=2))
            else:
                print(f"Serialized component is not a dictionary: {component_dict}")

            # Add component status information
            print("\nStatus Information:")
            status = config.get_component_status(args.component)
            for key, value in status.items():
                if key != "name" and key != "type":
                    print(f"  {key}: {value}")
        else:
            print(f"Component {args.component} not found")
        return

    # Default: show basic configuration information
    if not any(
        [
            args.validate,
            args.export,
            args.component,
            args.sources,
            args.profile,
            args.status,
            args.set,
        ]
    ):
        print("Word Forge Configuration")
        print("=======================")
        print(f"Version: {'.'.join(str(v) for v in config.version)}")
        print(
            f"Accessed components: {', '.join(sorted(config.get_accessed_components())) or 'None'}"
        )
        print("\nAvailable Components:")
        for component in sorted(config.get_available_components()):
            print(f"  • {component}")

        print("\nFor detailed information, use --component COMPONENT")
        print("For validation, use --validate")
        print("For export, use --export FILENAME")
        print("For value sources, use --sources")
        print("For component status, use --status")
        print("For profile application, use --profile NAME")
        print("For setting values, use --set COMPONENT SETTING VALUE")


if __name__ == "__main__":
    main()
