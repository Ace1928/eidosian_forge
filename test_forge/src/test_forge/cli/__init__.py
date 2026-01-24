from eidosian_core import eidosian
"""
Test_forge CLI - Minimal CLI interface.
"""
import argparse
import sys
from typing import Optional, List

@eidosian()
def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="test-forge",
        description="Test_forge - Forge component",
    )
    parser.add_argument("--version", action="store_true", help="Show version")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    subparsers.add_parser("status", help="Show forge status")
    
    args = parser.parse_args(argv)
    
    if args.version:
        print("Test_forge v0.1.0")
        return 0
    
    if args.command == "status":
        print("Test_forge Status: operational")
        return 0
    
    parser.print_help()
    return 0

app = main

if __name__ == "__main__":
    sys.exit(main())
