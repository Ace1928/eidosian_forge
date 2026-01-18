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
from .core.types import AnalysisResult, TransformationResult, RefactorOptions
from .analyzer import analyze_code
from .reporter import print_analysis_report
from .transformer import transform_code
from .transformer.filesystem import generate_files

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
        # Convert CLI args to options object
        options = RefactorOptions(
            source_path=parsed_args.source,
            output_dir=parsed_args.output_dir,
            package_name=parsed_args.package_name,
            analyze_only=parsed_args.analyze_only,
            dry_run=parsed_args.dry_run,
            verbose=parsed_args.verbose
        )
        
        # Execute the refactoring process
        result = execute_refactoring(options, clean=parsed_args.clean)
        return 0 if result["success"] else 1
        
    except Exception as e:
        print(f"Error: {e}")
        if parsed_args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def execute_refactoring(
    options: RefactorOptions, 
    clean: bool = False
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
    source_path = Path(options.source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")
    
    print(f"🔍 Analyzing {source_path}...")
    analysis_results = analyze_code(source_path)
    
    if options.analyze_only or options.dry_run or options.verbose:
        print_analysis_report(analysis_results)
        
    if options.analyze_only:
        return {"success": True, "analysis": analysis_results}
    
    print(f"🔮 Transforming {source_path}...")
    transform_results = transform_code(
        analysis_results, 
        output_dir=options.output_dir, 
        package_name=options.package_name
    )
    
    if clean:
        from .transformer.filesystem import clean_output_directory
        print(f"🧹 Cleaning output directory...")
        clean_output_directory(
            transform_results["output_path"], 
            dry_run=options.dry_run
        )
    
    # Generate files
    if not options.dry_run:
        print(f"📝 Generating files...")
        generated_files = generate_files(transform_results)
        print(f"✨ Generated {len(generated_files)} files")
        
        if options.verbose:
            for file_path in generated_files:
                print(f"  - {file_path}")
        
        print(f"✅ Transformation complete! Results in {transform_results['output_path']}")
    else:
        print("📋 Dry run completed - no files were modified")
    
    return {
        "success": True, 
        "analysis": analysis_results, 
        "transformation": transform_results
    }


def refactor(
    source_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    package_name: Optional[str] = None,
    analyze_only: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
    clean: bool = False
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
    options = RefactorOptions(
        source_path=source_path,
        output_dir=output_dir,
        package_name=package_name,
        analyze_only=analyze_only,
        dry_run=dry_run,
        verbose=verbose
    )
    
    return execute_refactoring(options, clean=clean)


if __name__ == "__main__":
    sys.exit(main())
