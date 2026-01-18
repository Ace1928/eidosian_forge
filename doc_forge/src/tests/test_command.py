#!/usr/bin/env python3
# 🌀 Eidosian Test Command System
"""
Test Command System - Command-line interface for test operations

This module provides command-line subcommands for test-related operations,
following Eidosian principles of structured control and contextual integrity.
"""
import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Any

# 📊 Self-aware logging system
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s)"
)
logger = logging.getLogger("doc_forge.test_command")

def add_test_subparsers(subparsers: Any) -> None:
    """
    Add test-related subcommands to the parser.
    
    Args:
        subparsers: Subparser collection to add commands to
    """
    # Analyze test coverage
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze test coverage"
    )
    analyze_parser.add_argument(
        "--repo-dir", type=Path, 
        help="Repository directory (default: auto-detect)"
    )
    analyze_parser.add_argument(
        "--output", "-o", type=Path,
        help="Output directory (default: tests directory)"
    )
    analyze_parser.set_defaults(func=cmd_analyze)
    
    # Generate test TODO
    todo_parser = subparsers.add_parser(
        "todo", help="Generate test TODO document"
    )
    todo_parser.add_argument(
        "--repo-dir", type=Path, 
        help="Repository directory (default: auto-detect)"
    )
    todo_parser.add_argument(
        "--output", "-o", type=Path,
        help="Output file (default: tests/TODO.md)"
    )
    todo_parser.set_defaults(func=cmd_todo)
    
    # Generate test stubs
    stubs_parser = subparsers.add_parser(
        "stubs", help="Generate test stubs"
    )
    stubs_parser.add_argument(
        "--repo-dir", type=Path, 
        help="Repository directory (default: auto-detect)"
    )
    stubs_parser.add_argument(
        "--output", "-o", type=Path,
        help="Output directory (default: tests directory)"
    )
    stubs_parser.set_defaults(func=cmd_stubs)
    
    # Generate test suite
    suite_parser = subparsers.add_parser(
        "suite", help="Generate test suite"
    )
    suite_parser.add_argument(
        "--repo-dir", type=Path, 
        help="Repository directory (default: auto-detect)"
    )
    suite_parser.add_argument(
        "--output", "-o", type=Path,
        help="Output directory (default: tests/generated_suite)"
    )
    suite_parser.set_defaults(func=cmd_suite)
    
    # Run tests
    run_parser = subparsers.add_parser(
        "run", help="Run tests"
    )
    run_parser.add_argument(
        "--repo-dir", type=Path, 
        help="Repository directory (default: auto-detect)"
    )
    run_parser.add_argument(
        "--pattern", type=str, default="test_*.py",
        help="Pattern for test files (default: test_*.py)"
    )
    run_parser.add_argument(
        "--pytest", action="store_true",
        help="Use pytest instead of unittest"
    )
    run_parser.set_defaults(func=cmd_run)

    # Basic test command
    basic_parser = subparsers.add_parser("basic", help="Run basic tests")
    basic_parser.add_argument("test_target", help="Target to test")
    basic_parser.set_defaults(func=run_basic_test)
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate documentation")
    validate_parser.add_argument("validate_target", help="Target to validate")
    validate_parser.add_argument("--level", "-l", choices=["basic", "full"], default="basic", 
                               help="Validation level")
    validate_parser.set_defaults(func=run_validate_test)

def cmd_analyze(args: argparse.Namespace) -> int:
    """
    Analyze test coverage and generate comprehensive report.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code (0 for success)
    """
    try:
        from ..doc_forge.ast_scanner import CodeEntityAnalyzer
        
        repo_root = args.repo_dir or Path.cwd()
        analyzer = CodeEntityAnalyzer(repo_root)
        
        output_dir = args.output or (repo_root / "tests")
        reports = analyzer.generate_comprehensive_report(output_dir)
        
        logger.info(f"✨ Generated Eidosian test analysis suite:")
        logger.info(f"📋 TODO document: {reports['todo']}")
        logger.info(f"📊 Coverage report: {reports['coverage']}")
        logger.info(f"🧪 Test stubs directory: {reports['stubs']}")
        logger.info(f"📈 Visualization: {reports['visualization']}")
        
        return 0
    except ImportError:
        logger.error("❌ Could not import ast_scanner module.")
        logger.error("💡 Make sure you're running from the project root directory.")
        return 1
    except Exception as e:
        logger.error(f"❌ Test analysis failed: {e}")
        if logging.getLogger().level <= logging.DEBUG:
            import traceback
            logger.debug(traceback.format_exc())
        return 1

def cmd_todo(args: argparse.Namespace) -> int:
    """
    Generate test TODO document.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code (0 for success)
    """
    try:
        from ..doc_forge.ast_scanner import CodeEntityAnalyzer
        
        repo_root = args.repo_dir or Path.cwd()
        analyzer = CodeEntityAnalyzer(repo_root)
        
        # Generate TODO document
        output_file = args.output or (repo_root / "tests" / "TODO.md")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Discover all code structures
        analyzer.discover_all_structures()
        
        # Generate TODO document
        todo_file = analyzer.generate_comprehensive_report(output_file.parent)["todo"]
        
        logger.info(f"✅ Generated test TODO document at: {todo_file}")
        return 0
    except ImportError:
        logger.error("❌ Could not import ast_scanner module.")
        logger.error("💡 Make sure you're running from the project root directory.")
        return 1
    except Exception as e:
        logger.error(f"❌ Test TODO generation failed: {e}")
        if logging.getLogger().level <= logging.DEBUG:
            import traceback
            logger.debug(traceback.format_exc())
        return 1

def cmd_stubs(args: argparse.Namespace) -> int:
    """
    Generate test stubs.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code (0 for success)
    """
    try:
        from ..doc_forge.ast_scanner import CodeEntityAnalyzer
        
        repo_root = args.repo_dir or Path.cwd()
        analyzer = CodeEntityAnalyzer(repo_root)
        
        # Generate stubs
        output_dir = args.output or (repo_root / "tests")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        stubs_info = analyzer.generate_test_stubs(output_dir)
        
        logger.info(f"✅ Generated {stubs_info['stub_count']} test stub files covering {stubs_info['covered_items']} code items")
        return 0
    except ImportError:
        logger.error("❌ Could not import ast_scanner module.")
        logger.error("💡 Make sure you're running from the project root directory.")
        return 1
    except Exception as e:
        logger.error(f"❌ Test stub generation failed: {e}")
        if logging.getLogger().level <= logging.DEBUG:
            import traceback
            logger.debug(traceback.format_exc())
        return 1

def cmd_suite(args: argparse.Namespace) -> int:
    """
    Generate test suite.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code (0 for success)
    """
    try:
        from ..doc_forge.ast_scanner import CodeEntityAnalyzer
        
        repo_root = args.repo_dir or Path.cwd()
        analyzer = CodeEntityAnalyzer(repo_root)
        
        # Generate suite
        output_dir = args.output or (repo_root / "tests" / "generated_suite")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        suite_dir = analyzer.generate_test_suite(output_dir)
        
        logger.info(f"✅ Generated test suite at: {suite_dir}")
        return 0
    except ImportError:
        logger.error("❌ Could not import ast_scanner module.")
        logger.error("💡 Make sure you're running from the project root directory.")
        return 1
    except Exception as e:
        logger.error(f"❌ Test suite generation failed: {e}")
        if logging.getLogger().level <= logging.DEBUG:
            import traceback
            logger.debug(traceback.format_exc())
        return 1

def cmd_run(args: argparse.Namespace) -> int:
    """
    Run tests.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code (0 for success)
    """
    repo_root = args.repo_dir or Path.cwd()
    tests_dir = repo_root / "tests"
    
    try:
        if args.pytest:
            # Run tests with pytest
            import pytest
            
            logger.info(f"🧪 Running tests with pytest")
            result = pytest.main([str(tests_dir), "-xvs", "--pattern", args.pattern])
            
            return 0 if result == 0 else 1
        else:
            # Run tests with unittest
            import unittest
            
            logger.info(f"🧪 Running tests with unittest")
            
            # Change to the repository root so imports work correctly
            original_dir = Path.cwd()
            os.chdir(repo_root)
            
            try:
                test_loader = unittest.TestLoader()
                test_suite = test_loader.discover(str(tests_dir), pattern=args.pattern)
                
                test_runner = unittest.TextTestRunner(verbosity=2)
                result = test_runner.run(test_suite)
                
                return 0 if result.wasSuccessful() else 1
            finally:
                # Change back to the original directory
                os.chdir(original_dir)
    except ImportError as e:
        logger.error(f"❌ Could not import test framework: {e}")
        return 1
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        if logging.getLogger().level <= logging.DEBUG:
            import traceback
            logger.debug(traceback.format_exc())
        return 1

def run_basic_test(args: argparse.Namespace) -> int:
    """
    Run basic test functionality.
    
    Args:
        args: Command line arguments
        
    Returns:
        int: Exit code (0 for success)
    """
    print(f"Running basic test on target: {args.test_target}")
    return 0

def run_validate_test(args: argparse.Namespace) -> int:
    """
    Run validation test.
    
    Args:
        args: Command line arguments
        
    Returns:
        int: Exit code (0 for success)
    """
    print(f"Validating: {args.validate_target}")
    print(f"Validation level: {args.level}")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Doc Forge Test Command System")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    add_test_subparsers(subparsers)
    
    args = parser.parse_args()
    
    if hasattr(args, "func"):
        sys.exit(args.func(args))
    else:
        parser.print_help()
        sys.exit(1)
