#!/usr/bin/env python3
"""
Command-line interface for Eidosian Repo Forge.

This module provides the entry point for creating universal Eidosian monorepo
structures from the command line, with full customization options.
"""
import argparse
import sys
import logging
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

# Fix for running directly vs as part of package
try:
    from . import __version__
    from .core.directory import create_directory_structure
    from .generators.docs import create_documentation_structure
    from .generators.project import create_project_scaffold
    from .generators.config import create_configuration_files, create_eidosian_json
    from .generators.scripts import create_script_files
    from .core.diagnostics import diagnostics, verify_json_serializable, safe_int
except ImportError:
    # When running directly, add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from repo_forge import __version__
    from repo_forge.core.directory import create_directory_structure
    from repo_forge.generators.docs import create_documentation_structure
    from repo_forge.generators.project import create_project_scaffold
    from repo_forge.generators.config import create_configuration_files, create_eidosian_json
    from repo_forge.generators.scripts import create_script_files
    from repo_forge.core.diagnostics import diagnostics, verify_json_serializable, safe_int


def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging with appropriate verbosity level.
    
    Sets up a consistent logging format with colorized output when available,
    optimizing for both human readability and machine parsing.
    
    Args:
        verbose: Whether to enable debug-level logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    
    # Try to use colorized logging if colorama is available
    try:
        import colorama
        colorama.init()
        
        # Define ANSI color codes
        colors = {
            'DEBUG': '\033[36m',  # Cyan
            'INFO': '\033[32m',   # Green
            'WARNING': '\033[33m', # Yellow
            'ERROR': '\033[31m',  # Red
            'CRITICAL': '\033[41m', # Red background
            'RESET': '\033[0m'    # Reset
        }
        
        class ColorFormatter(logging.Formatter):
            def format(self, record):
                levelname = record.levelname
                if levelname in colors:
                    levelname_color = colors[levelname] + levelname + colors['RESET']
                    record.levelname = levelname_color
                return super().format(record)
        
        formatter = ColorFormatter("%(levelname)s: %(message)s")
    except ImportError:
        formatter = logging.Formatter("%(levelname)s: %(message)s")
    
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for hdlr in root_logger.handlers:
        root_logger.removeHandler(hdlr)
    
    root_logger.addHandler(handler)


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="🔮 Eidosian Repo Forge - Universal Monorepo Creator",
        epilog="A meticulously engineered toolkit for creating perfect repository structures."
    )
    
    parser.add_argument(
        "-o", "--output", 
        type=Path, 
        default=Path.cwd(),
        help="Directory where the repository will be created"
    )
    
    parser.add_argument(
        "-n", "--name",
        type=str,
        default="eidosian_monorepo",
        help="Name of the repository"
    )
    
    parser.add_argument(
        "--languages", 
        nargs="+", 
        default=["python", "nodejs", "go", "rust"],
        help="Programming languages to support"
    )
    
    parser.add_argument(
        "--init-git", 
        action="store_true",
        help="Initialize git repository"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--version", 
        action="version",
        version=f"%(prog)s {__version__}"
    )
    
    parser.add_argument(
        "--safe-mode",
        action="store_true",
        help="Don't overwrite existing files"
    )
    
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Bootstrap the repo_forge package itself with safe defaults"
    )
    
    return parser.parse_args(args)


def process_generator_result(result: Dict[str, Any], operation_name: str) -> bool:
    """
    Process and validate result from generator functions with extensive diagnostics.
    
    Args:
        result: Result dictionary from generator function
        operation_name: Name of the operation for logging
        
    Returns:
        True if successful, False otherwise
    """
    if not result.get("success", False):
        error = result.get("error", "Unknown error")
        logging.error(f"❌ Error in {operation_name}: {error}")
        return False
    
    # 🧬 EIDOSIAN TYPE INTEGRITY SYSTEM
    # Exhaustive validation with surgical precision
    diagnostics.trace(f"Processing generator result for {operation_name}", result)
    
    # CRITICAL FIX: Validate and sanitize count field with comprehensive diagnostics
    if "count" in result:
        try:
            # Capture original state for diagnostics
            original_count = result["count"]
            original_type = type(original_count).__name__
            
            # Explicit primitive int conversion with diagnostic verification
            result["count"] = safe_int(result["count"])
            
            # Log the conversion for transparency
            diagnostics.logger.debug(
                f"Count field converted: {original_count} ({original_type}) -> "
                f"{result['count']} ({type(result['count']).__name__})"
            )
        except (TypeError, ValueError) as e:
            # Implement self-healing recovery with detailed logging
            diagnostics.logger.warning(
                f"⚠️ Count field ({result.get('count')}) conversion failed: {e}. "
                f"Falling back to created_files length."
            )
            result["count"] = len(result.get("created_files", []))
    else:
        # If count is missing, derive it from created_files with validation
        file_count = len(result.get("created_files", []))
        result["count"] = safe_int(file_count)
        diagnostics.logger.debug(f"Count field was missing, derived from created_files: {result['count']}")
    
    # Verify JSON serialization before returning
    try:
        # Ultra-defensive serialization check with comprehensive error trapping
        verify_json_serializable(result, operation_name)
    except ValueError as e:
        # Last-resort emergency fix if serialization still fails
        diagnostics.logger.error(f"💥 Emergency repair needed for {operation_name} result: {e}")
        
        # Create guaranteed-serializable sanitized copy
        sanitized = {
            "success": bool(result.get("success")),
            "count": safe_int(result.get("count", 0)),
            "created_files": list(str(f) for f in result.get("created_files", [])),
            "base_path": str(result.get("base_path", "")),
            "_emergency_sanitized": True
        }
        
        # Replace original with sanitized version
        result.clear()
        result.update(sanitized)
        
        diagnostics.logger.info("🛠️ Emergency sanitization complete. Result is now serializable.")
        
    return True

def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for the CLI.
    
    Args:
        args: Command line arguments
        
    Returns:
        Exit code (0 for success)
    """
    parsed_args = parse_args(args)
    setup_logging(parsed_args.verbose)
    
    try:
        # Special handling for bootstrapping repo_forge itself
        if parsed_args.bootstrap:
            from pathlib import Path
            import os
            
            # Get the repo_forge package root directory
            package_dir = Path(__file__).parent
            repo_root = package_dir.parent
            
            logging.info("🔄 Bootstrapping repo_forge with its own structure...")
            logging.info("📍 Detected package at: %s", package_dir)
            logging.info("📍 Repository root at: %s", repo_root)
            
            # Force safe mode when bootstrapping to avoid overwriting existing files
            overwrite = False
            repo_path = repo_root
            
            # Create core directory structure (preserving existing files)
            logging.info("🌟 Creating Universal Eidosian Monorepo Structure...")
            create_directory_structure(repo_path, parsed_args.languages)
            
            # Create documentation structure aligned with universal standard
            logging.info("📚 Generating comprehensive documentation structure...")
            create_documentation_structure(repo_path, overwrite=overwrite)
            
            # Create configuration files with safe mode
            logging.info("⚙️ Creating configuration files...")
            create_configuration_files(
                repo_path, 
                "repo_forge", 
                parsed_args.languages, 
                overwrite=overwrite
            )
            
            # Create script files with safe mode
            logging.info("🛠️ Generating script files...")
            script_result = create_script_files(
                repo_path, 
                parsed_args.languages, 
                overwrite=overwrite
            )
            
            if not process_generator_result(script_result, "script generation"):
                logging.error("❌ Script generation failed")
                return 1
            
            # Create project scaffolds with safe mode
            logging.info("🏗️ Scaffolding project structures...")
            for language in parsed_args.languages:
                create_project_scaffold(repo_path, language, overwrite=overwrite)
            
            # Generate .eidosian.json metadata file
            logging.info("🔮 Creating .eidosian.json metadata...")
            create_eidosian_json(repo_path, "repo_forge")
            
            logging.info("✅ Repository structure has been bootstrapped.")
            logging.info("🚀 Ready for infinite scalability and multi-language development.")
            
            return 0
        else:
            # Regular mode - create new repo
            overwrite = not parsed_args.safe_mode  # Invert safe_mode for clarity
            repo_path = parsed_args.output / parsed_args.name
        
            # Create the base directory if it doesn't exist
            repo_path.mkdir(parents=True, exist_ok=True)
            
            # Standard repo creation process
            # Create core directory structure
            logging.info("🌟 Creating Universal Eidosian Monorepo Structure...")
            dir_result = create_directory_structure(repo_path, parsed_args.languages) or {"success": True, "count": 0}
            if not process_generator_result(dir_result, "directory structure"):
                return 1

            logging.info("📚 Generating comprehensive documentation structure...")
            doc_result = create_documentation_structure(repo_path) or {"success": True, "count": 0}
            if not process_generator_result(doc_result, "documentation structure"):
                return 1

            logging.info("⚙️ Creating configuration files...")
            config_result = create_configuration_files(repo_path, parsed_args.name, parsed_args.languages) or {"success": True, "count": 0}
            if not process_generator_result(config_result, "configuration files"):
                return 1

            logging.info("🛠️ Generating script files...")
            script_result = create_script_files(repo_path, parsed_args.languages, overwrite)
            if not process_generator_result(script_result, "script generation"):
                logging.error("❌ Script generation failed")
                return 1

            logging.info("🏗️ Scaffolding project structures...")
            project_result = create_project_scaffold(repo_path, parsed_args.languages, overwrite) or {"success": True, "count": 0}
            if not process_generator_result(project_result, "project scaffolding"):
                return 1
            
            # Initialize git repository if requested
            if parsed_args.init_git:
                import subprocess
                logging.info("🔄 Initializing git repository...")
                subprocess.run(["git", "init"], cwd=repo_path, check=True)
            
            logging.info("✅ Universal Eidosian Monorepo structure has been created.")
            logging.info("🚀 Ready for infinite scalability and multi-language development.")
            
            return 0
        
    except Exception as e:
        logging.error(f"❌ Error: {e}")
        if parsed_args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())