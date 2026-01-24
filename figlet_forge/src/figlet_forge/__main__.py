from eidosian_core import eidosian
"""
Main entry point for the figlet_forge package when executed as a module.

This allows executing the package directly with:
  python -m figlet_forge [arguments]
"""

import sys
from typing import List, Optional

from figlet_forge.cli.main import main


@eidosian()
def entry_point(args: Optional[List[str]] = None) -> int:
    """
    Main entry point function for module execution.

    Args:
        args: Command line arguments (defaults to sys.argv[1:])

    Returns:
        Exit code
    """
    if args is None:
        args = sys.argv[1:]
    return main(args)


if __name__ == "__main__":
    sys.exit(entry_point())
