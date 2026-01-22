"""
Command line interface for metadata_forge.

This module provides the CLI for working with Eidosian metadata.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .schema import create_metadata_template, validate_metadata


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for the CLI.
    
    Args:
        args: Command line arguments
        
    Returns:
        Exit code (0 for success)
    """
    parser = argparse.ArgumentParser(
        description="🔮 Eidosian Metadata Forge CLI"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Generate template command
    gen_parser = subparsers.add_parser(
        "template", 
        help="Generate a metadata template"
    )
    gen_parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output file path (use - for stdout)"
    )
    
    # Validate metadata command  
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate metadata"
    )
    validate_parser.add_argument(
        "file",
        type=Path,
        help="Metadata file to validate"
    )
    
    # Version command
    version_parser = subparsers.add_parser(
        "version",
        help="Show version information"
    )
    
    parsed_args = parser.parse_args(args)
    
    if parsed_args.command == "template":
        template = create_metadata_template()
        json_str = json.dumps(template, indent=2)
        
        if not parsed_args.output or str(parsed_args.output) == "-":
            print(json_str)
        else:
            with open(parsed_args.output, "w") as f:
                f.write(json_str)
            print(f"Template written to {parsed_args.output}")
            
    elif parsed_args.command == "validate":
        try:
            with open(parsed_args.file) as f:
                metadata = json.load(f)
            
            errors = validate_metadata(metadata)
            if errors:
                print("❌ Validation errors:")
                for field, field_errors in errors.items():
                    print(f"  Field '{field}':")
                    for error in field_errors:
                        print(f"    - {error}")
                return 1
            else:
                print("✅ Metadata is valid")
                return 0
        except Exception as e:
            print(f"Error: {e}")
            return 1
            
    elif parsed_args.command == "version":
        from . import __version__
        print(f"metadata_forge v{__version__}")
        
    else:
        parser.print_help()
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
