"""
Type Forge CLI - Schema registry and validation CLI.
"""
import argparse
import json
import sys
from typing import Optional, List

def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="type-forge",
        description="🔧 Type Forge - Schema registry and validation",
    )
    parser.add_argument("--version", action="store_true", help="Show version")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Register schema
    reg_parser = subparsers.add_parser("register", help="Register a schema")
    reg_parser.add_argument("name", help="Schema name")
    reg_parser.add_argument("schema", help="Schema JSON (or @filename)")
    
    # Validate data
    val_parser = subparsers.add_parser("validate", help="Validate data against schema")
    val_parser.add_argument("schema_name", help="Schema name")
    val_parser.add_argument("data", help="Data JSON (or @filename)")
    
    # List schemas
    subparsers.add_parser("list", help="List registered schemas")
    
    # Status
    subparsers.add_parser("status", help="Show type forge status")
    
    args = parser.parse_args(argv)
    
    if args.version:
        print("Type Forge v0.1.0")
        return 0
    
    if not args.command:
        parser.print_help()
        return 0
    
    from type_forge import TypeCore
    core = TypeCore()
    
    if args.command == "status":
        print("🔧 Type Forge Status")
        print(f"  Registered schemas: {len(core.snapshot())}")
        return 0
    
    elif args.command == "list":
        schemas = core.snapshot()
        if not schemas:
            print("No schemas registered")
        for name in schemas:
            print(f"  {name}")
        return 0
    
    elif args.command == "register":
        schema_str = args.schema
        if schema_str.startswith("@"):
            with open(schema_str[1:]) as f:
                schema = json.load(f)
        else:
            schema = json.loads(schema_str)
        core.register_schema(args.name, schema)
        print(f"✓ Registered schema: {args.name}")
        return 0
    
    elif args.command == "validate":
        data_str = args.data
        if data_str.startswith("@"):
            with open(data_str[1:]) as f:
                data = json.load(f)
        else:
            data = json.loads(data_str)
        try:
            core.validate(args.schema_name, data)
            print("✓ Validation passed")
            return 0
        except Exception as e:
            print(f"✗ Validation failed: {e}")
            return 1
    
    return 0

app = main

if __name__ == "__main__":
    sys.exit(main())
