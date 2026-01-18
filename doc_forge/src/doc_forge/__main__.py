#!/usr/bin/env python3
# 🌀 Eidosian Module Entry Point
"""
Module Entry Point - For Direct Module Execution

This module serves as the entry point when Doc Forge is executed
directly as a module using 'python -m doc_forge'.

Following Eidosian principles of:
- Velocity as Intelligence: Direct path to execution
- Structure as Control: Clear entry point architecture 
- Self-Awareness: Understanding execution context
- Precision as Style: Elegant error handling with informative messages

Examples:
    Execute the module directly:

    .. code-block:: bash

        $ python -m doc_forge

    With specific arguments:

    .. code-block:: bash

        $ python -m doc_forge build --clean
"""

import sys
import os
import logging

# Configure logging - the sentient nervous system of our architecture
logging.basicConfig(
    level=logging.INFO if not os.environ.get("DOC_FORGE_DEBUG") else logging.DEBUG,
    format="%(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s)"
)
logger = logging.getLogger("doc_forge.__main__")

try:
    # Import version info first - our temporal anchor
    from .version import get_version_string
    
    # Then try to import our main runner - the heart of our operation
    try:
        from .main import main
    except ImportError:
        # Fallback chain - like a master finding alternative paths when one is blocked
        try:
            from .doc_forge import main
        except ImportError:
            # Ultimate fallback - the final sanctuary
            from .__init__ import main as run_cli
            main = run_cli
            
    logger.debug(f"🌀 Doc Forge v{get_version_string()} module entry point activated")
    
except ImportError as e:
    # Even in failure, maintain elegance and wisdom
    print(f"🌋 Critical initialization error: {str(e)}")
    print("💠 Ensure Doc Forge is properly installed or run from source directory.")
    sys.exit(1)

def module_entry_point() -> int:
    """
    Dedicated function for the module entry point with pristine error handling.
    
    This serves as a guardian of execution, catching exceptions with grace
    and providing insightful feedback even in failure.
    
    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    try:
        logger.debug("🚀 Invoking main execution path")
        return main()
    except KeyboardInterrupt:
        logger.info("⌨️  Process interrupted by user - exiting with dignity")
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        logger.error(f"💥 Execution failed: {str(e)}", exc_info=True)
        print(f"🔮 Doc Forge encountered an unexpected situation: {str(e)}")
        print("📜 Check logs for details or run with DOC_FORGE_DEBUG=1 for verbose output")
        return 1

if __name__ == "__main__":
    # Display version banner - our herald announcing our presence!
    try:
        print(f"🌀 Doc Forge v{get_version_string()} - Eidosian Documentation System")
        print("✨ Crafting documentation with precision, structure, flow, and self-awareness")
    except NameError:
        # Even without version info, we maintain composure
        print("🌀 Doc Forge - Eidosian Documentation System")
    
    # Pass control to our entry point with elegant error capture
    sys.exit(module_entry_point())
