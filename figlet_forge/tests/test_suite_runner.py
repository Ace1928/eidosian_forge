#!/usr/bin/env python3
"""
Comprehensive test suite runner for Figlet Forge.

This script provides a unified interface for running all tests with
detailed reporting, coverage analysis, and options for selective testing.
Following Eidosian principles of structure, clarity, and precision.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Set

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class FigletTestRunner:
    """Runner for the Figlet Forge test suite with comprehensive features."""

    def __init__(self, options: argparse.Namespace) -> None:
        """
        Initialize the test runner with command line options.

        Args:
            options: Command line argument namespace
        """
        self.options = options
        self.project_root = Path(__file__).parent.parent
        self.start_time = time.time()

        # Set up output format based on options
        self.use_color = not options.no_color and sys.stdout.isatty()

    def print_header(self, text: str) -> None:
        """
        Print a formatted header.

        Args:
            text: Header text to display
        """
        if self.use_color:
            print(f"\n\033[1;36m{'═' * 80}\033[0m")
            print(f"\033[1;36m {text}\033[0m")
            print(f"\033[1;36m{'═' * 80}\033[0m\n")
        else:
            print(f"\n{'═' * 80}")
            print(f" {text}")
            print(f"{'═' * 80}\n")

    def print_success(self, text: str) -> None:
        """
        Print a success message.

        Args:
            text: Success message to display
        """
        if self.use_color:
            print(f"\033[32m✓ {text}\033[0m")
        else:
            print(f"PASS: {text}")

    def print_failure(self, text: str) -> None:
        """
        Print a failure message.

        Args:
            text: Failure message to display
        """
        if self.use_color:
            print(f"\033[31m✗ {text}\033[0m")
        else:
            print(f"FAIL: {text}")

    def print_info(self, text: str) -> None:
        """
        Print an informational message.

        Args:
            text: Informational message to display
        """
        if self.use_color:
            print(f"\033[1;34m{text}\033[0m")
        else:
            print(text)

    def discover_tests(self) -> Dict[str, List[Path]]:
        """
        Discover all test files in the project.

        Returns:
            Dictionary mapping test categories to lists of test files
        """
        tests_dir = self.project_root / "tests"

        # Define test categories
        categories = {
            "unit": tests_dir / "unit",
            "integration": tests_dir / "integration",
            "compat": tests_dir / "compat",
            "fonts": tests_dir / "fonts",
        }

        # Discover tests in each category
        discovered: Dict[str, List[Path]] = {}
        for category, directory in categories.items():
            if directory.exists():
                discovered[category] = list(directory.glob("test_*.py"))
                # Add test files in the root of the category directory
                discovered[category].extend(directory.glob("*.py"))

        # Add tests in the root directory
        if tests_dir.exists():
            discovered["root"] = list(tests_dir.glob("test_*.py"))

        return discovered

    def run_pytest(self, test_files: List[Path], extra_args: List[str] = None) -> bool:
        """
        Run pytest on specified test files.

        Args:
            test_files: List of test files to run
            extra_args: Additional pytest arguments

        Returns:
            True if tests passed, False if any failed
        """
        if not test_files:
            return True

        # Build command
        cmd = [sys.executable, "-m", "pytest"]

        # Add coverage if requested
        if self.options.coverage:
            cmd.extend(["--cov=figlet_forge", "--cov-report=term"])

            # Add HTML report if requested
            if self.options.html_coverage:
                cmd.append("--cov-report=html")

        # Add verbosity
        if self.options.verbosity > 0:
            cmd.append(f"-{'v' * self.options.verbosity}")

        # Add extra args if provided
        if extra_args:
            cmd.extend(extra_args)

        # Add test files
        cmd.extend(str(f) for f in test_files)

        # Run the command
        self.print_info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)

        return result.returncode == 0

    def run_unittest(self, test_files: List[Path]) -> bool:
        """
        Run unittest on specified test files.

        Args:
            test_files: List of test files to run

        Returns:
            True if tests passed, False if any failed
        """
        if not test_files:
            return True

        # Build command
        cmd = [sys.executable, "-m", "unittest"]

        # Add test files
        cmd.extend(str(f) for f in test_files)

        # Run the command
        self.print_info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)

        return result.returncode == 0

    def run_type_checking(self) -> bool:
        """
        Run static type checking with mypy.

        Returns:
            True if type checking passed, False if failed
        """
        try:
            import mypy.api
        except ImportError:
            self.print_info("Mypy not installed, skipping type checking")
            return True

        self.print_info("Running type checking with mypy...")

        # Configure mypy options
        mypy_args = [
            str(self.project_root / "src"),
            "--ignore-missing-imports",
            "--disallow-untyped-defs",
            "--disallow-incomplete-defs",
            "--check-untyped-defs",
        ]

        # Run mypy
        stdout, stderr, exit_code = mypy.api.run(mypy_args)

        if exit_code:
            self.print_failure("Type checking failed")
            print(stdout)
            print(stderr)
            return False
        else:
            self.print_success("Type checking passed")
            return True

    def run_test_suite(self) -> bool:
        """
        Run the full test suite based on options.

        Returns:
            True if all selected tests passed, False otherwise
        """
        # Discover all tests
        discovered_tests = self.discover_tests()

        # Filter categories based on options
        categories_to_run: Set[str] = set()
        if self.options.unit:
            categories_to_run.add("unit")
        if self.options.integration:
            categories_to_run.add("integration")
        if self.options.compat:
            categories_to_run.add("compat")
        if self.options.fonts:
            categories_to_run.add("fonts")
        if self.options.all or not categories_to_run:
            # Run all categories if none specified or --all used
            categories_to_run = set(discovered_tests.keys())

        # Track results
        all_passed = True

        # Run type checking if requested
        if self.options.typecheck:
            type_check_passed = self.run_type_checking()
            all_passed = all_passed and type_check_passed

        # Run tests for each category
        for category in categories_to_run:
            if category not in discovered_tests:
                self.print_info(f"No tests found for category: {category}")
                continue

            test_files = discovered_tests[category]
            if not test_files:
                self.print_info(f"No test files found in category: {category}")
                continue

            # Print category header
            self.print_header(f"Running {category} tests")

            # Choose test runner based on options
            if self.options.unittest:
                passed = self.run_unittest(test_files)
            else:
                passed = self.run_pytest(test_files)

            # Update overall status
            all_passed = all_passed and passed

            # Print category result
            if passed:
                self.print_success(f"{category.upper()} TESTS PASSED")
            else:
                self.print_failure(f"{category.upper()} TESTS FAILED")

        # Print summary
        elapsed = time.time() - self.start_time
        self.print_header(f"Test Suite Complete (in {elapsed:.2f}s)")

        if all_passed:
            self.print_success("ALL TESTS PASSED")
            return True
        else:
            self.print_failure("SOME TESTS FAILED")
            return False


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Figlet Forge Test Suite Runner")

    # Test selection options
    selection = parser.add_argument_group("Test Selection")
    selection.add_argument("--all", action="store_true", help="Run all tests (default)")
    selection.add_argument("--unit", action="store_true", help="Run unit tests")
    selection.add_argument(
        "--integration", action="store_true", help="Run integration tests"
    )
    selection.add_argument(
        "--compat", action="store_true", help="Run compatibility tests"
    )
    selection.add_argument("--fonts", action="store_true", help="Run font tests")

    # Test runner options
    runner = parser.add_argument_group("Test Runner")
    runner.add_argument(
        "--unittest", action="store_true", help="Use unittest instead of pytest"
    )
    runner.add_argument(
        "--coverage", action="store_true", help="Generate test coverage report"
    )
    runner.add_argument(
        "--html-coverage", action="store_true", help="Generate HTML coverage report"
    )
    runner.add_argument(
        "--typecheck", action="store_true", help="Run static type checking with mypy"
    )

    # Output options
    output = parser.add_argument_group("Output Options")
    output.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        dest="verbosity",
        help="Increase verbosity (can be used multiple times)",
    )
    output.add_argument(
        "--no-color", action="store_true", help="Disable colored output"
    )

    return parser.parse_args()


def main() -> int:
    """
    Main entry point for the test suite runner.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    args = parse_args()
    runner = FigletTestRunner(args)

    try:
        success = runner.run_test_suite()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\nTest suite interrupted by user")
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        print(f"Error running test suite: {e}")
        if args.verbosity > 0:
            import traceback

            traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
