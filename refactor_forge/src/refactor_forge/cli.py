"""
Command Line Interface - User Interaction Layer 🖥️

Provides a streamlined CLI for the Eidosian Refactor tool, allowing users
to analyze and transform code from the command line with maximum efficiency.

<!-- VERSION_START -->
Version: 0.1.0
<!-- VERSION_END -->
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

from . import __version__
from .analyzer import analyze_code
from .reporter import print_analysis_report

# ╭──────────────────────────────────────────────────────╮
# │  🎮 Command Interface - Argument Processing          │
# ╰──────────────────────────────────────────────────────╯

def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments with precision and clarity.
    
    Args:
        args: Command line arguments (None uses sys.argv)
        
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description=f"Eidosian Refactor v{__version__} - Transform code into perfect modular architecture"
    )
    
    parser.add_argument(
        "source",
        help="Path to source Python file to analyze or refactor"
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        help="Output directory for the refactored package (default: derived from source)"
    )
    
    parser.add_argument(
        "-n", "--package-name",
        help="Name for the refactored package (default: derived from source filename)"
    )
    
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze the source, don't perform refactoring"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making any changes"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean output directory before generating files"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show version number and exit"
    )
    
    return parser.parse_args(args)

# ╭──────────────────────────────────────────────────────╮
# │  🚀 Main Entry Point - Execution Flow               │
# ╰──────────────────────────────────────────────────────╯

def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI - directs workflow with minimal friction.
    
    Args:
        args: Command line arguments (None uses sys.argv)
        
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    parsed_args = parse_args(args)
    
    try:
        result = execute_refactoring(
            parsed_args.source,
            output_dir=parsed_args.output_dir,
            analyze_only=parsed_args.analyze_only,
            dry_run=parsed_args.dry_run,
            verbose=parsed_args.verbose,
            clean=parsed_args.clean,
        )
        return 0 if result["success"] else 1
        
    except Exception as e:
        print(f"Error: {e}")
        if parsed_args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def execute_refactoring(
    source_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    analyze_only: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
    clean: bool = False,
) -> Dict[str, Any]:
    """Execute the refactoring process based on given options.
    
    Args:
        options: Refactoring options
        clean: Whether to clean output directory first
        
    Returns:
        Dict containing analysis and transformation results
        
    Raises:
        FileNotFoundError: If source file doesn't exist
    """
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")
    
    print(f"🔍 Analyzing {source_path}...")
    analysis_results = analyze_code(source_path)
    
    if analyze_only or dry_run or verbose:
        print_analysis_report(analysis_results)
        
    if analyze_only:
        return {"success": True, "analysis": analysis_results}

    print("⚠️ Transformation is not implemented in this minimal CLI.")
    if output_dir or clean:
        print("⚠️ Output directory options are ignored in this mode.")
    return {"success": True, "analysis": analysis_results}


def refactor(
    source_path: Union[str, Path],
    analyze_only: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
    clean: bool = False,
) -> Dict[str, Any]:
    """High-level API for programmatic refactoring - maintains full compatibility.
    
    Args:
        source_path: Path to source Python file
        output_dir: Output directory (default: derived from source)
        package_name: Package name (default: derived from source)
        analyze_only: Only perform analysis without transformation
        dry_run: Show what would be done without making changes
        verbose: Enable verbose output
        clean: Whether to clean output directory first
        
    Returns:
        Dict containing analysis and transformation results
        
    Raises:
        FileNotFoundError: If source file doesn't exist
    """
    return execute_refactoring(
        source_path,
        analyze_only=analyze_only,
        dry_run=dry_run,
        verbose=verbose,
        clean=clean,
    )


if __name__ == "__main__":
    sys.exit(main())
