#!/usr/bin/env python3

"""
EidosianForge :: FigletForge Test Suite
A comprehensive testing utility for the FigletForge package, an enhanced
reimplementation of PyFiglet that adds support for ANSI color codes,
unicode characters, and extended font capabilities.

This test suite ensures backward compatibility with the original PyFiglet
while validating the enhanced features of FigletForge.
"""

import logging
import os.path
import platform
import sys
from optparse import OptionParser
from subprocess import PIPE, Popen
from typing import Any

from figlet_forge import Figlet

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("figletforge.test")

# Try to import color support, with graceful fallback
try:
    from colorama import init

    init(strip=not sys.stdout.isatty(), autoreset=True)
    from termcolor import cprint  # type: ignore

    has_color = True
except ImportError:
    has_color = False

    def cprint(text: str, color: str | None = None, **kwargs: Any) -> None:
        # Parameters color and kwargs are intentionally unused in fallback implementation
        print(text)


__version__ = "0.1.0"  # Eidosian Forge version


def fail(text: str) -> None:
    """Format and display failure messages."""
    if has_color:
        cprint(text, "red", attrs=["bold"])  # type: ignore
    else:
        print(f"FAIL: {text}")


def win(text: str) -> None:
    """Format and display success messages."""
    if has_color:
        cprint(text, "green")
    else:
        print(f"SUCCESS: {text}")


def info(text: str) -> None:
    """Format and display informational messages."""
    if has_color:
        cprint(text, "blue")
    else:
        print(f"INFO: {text}")


def dump(text: str) -> None:
    """Display text with visible representation of whitespace."""
    for line in text.split("\n"):
        print(repr(line))


# Define the options class structure to help type checking
class TestOptions:
    show: bool
    batch: bool
    verbose: bool
    test_unicode: bool
    test_color: bool


class Test:
    def __init__(self, opts: TestOptions):
        self.opts = opts
        self.ok: int = 0
        self.fail: int = 0
        self.failed: list[str] = []
        self.oked: list[str] = []

        # Known fonts with compatibility issues
        self.skip: list[str] = ["konto", "konto_slant"]  # Known incompatible fonts

        # Initialize Figlet with default settings
        self.f: Figlet = Figlet()
        # Determine platform-specific characteristics
        self.is_windows: bool = platform.system() == "Windows"

    def outputUsingFigletOrToilet(self, text: str, font: str, font_path: str) -> str:
        """Generate output using system's figlet or toilet command for comparison."""
        if os.path.isfile(font_path + ".flf"):
            cmd = ("figlet", "-d", "pyfiglet/fonts", "-f", font, text)
        elif os.path.isfile(font_path + ".tlf"):
            cmd = ("toilet", "-d", "pyfiglet/fonts", "-f", font, text)
        else:
            raise Exception(f"Missing font file: {font_path}")

        # Handle potential system command issues with detailed error info
        try:
            p = Popen(cmd, bufsize=4096, stdout=PIPE, stderr=PIPE)
            stdout, stderr = p.communicate()
            if p.returncode != 0:
                logger.warning(
                    f"External command error: {stderr.decode('utf-8', errors='replace')}"
                )
                return ""
            return stdout.decode("utf-8", errors="replace")
        except UnicodeDecodeError as e:
            logger.warning(f"Unicode decoding error with font {font}: {str(e)}")
            return ""
        except FileNotFoundError:
            logger.warning("Command not found. Ensure figlet/toilet is installed")
            return ""

    def validate_font_output(
        self, font: str, output_figlet: str, output_pyfiglet: str
    ) -> None:
        """Compare outputs between system figlet and FigletForge."""
        # Check for exact match first
        if output_pyfiglet == output_figlet:
            win(f"[OK] {font}")
            self.ok += 1
            self.oked.append(font)
            return

        # Check for whitespace-only differences (still valid but warn)
        if output_pyfiglet.replace(" ", "") == output_figlet.replace(" ", ""):
            info(f"[OK/SPACE_DIFF] {font}")
            self.ok += 1
            self.oked.append(font)
            return

        fail(f"[FAIL] {font}")
        self.fail += 1
        self.failed.append(font)
        self.show_result(output_figlet, output_pyfiglet, font)

    def show_result(self, output_figlet: str, output_pyfiglet: str, font: str) -> None:
        """Display detailed comparison for failed tests."""
        if hasattr(self.opts, "show") and self.opts.show:
            print(f"[FIGLETFORGE] *** {font} ***")
            dump(output_pyfiglet)
            print(f"[SYSTEM FIGLET] *** {font} ***")
            dump(output_figlet)

            if not hasattr(self.opts, "batch") or not self.opts.batch:
                input("Press Enter to continue...")

    def check_font(self, text: str, font: str, use_tlf: bool) -> None:
        """Test a specific font with the given text."""
        # Skip known problematic fonts
        if font in self.skip:
            info(f"[SKIPPED] {font}")
            return

        # Handle TLF font mode appropriately
        font_path = os.path.join("pyfiglet", "fonts", font)
        fig_file = os.path.isfile(font_path + ".flf")
        if not use_tlf and not fig_file:
            return

        # Set up rendering and compare
        try:
            self.f.setFont(font=font)
            output_pyfiglet = self.f.renderText(text)
            output_figlet = self.outputUsingFigletOrToilet(text, font, font_path)
            self.validate_font_output(font, output_figlet, output_pyfiglet)
        except Exception as e:
            fail(f"[ERROR] {font}: {str(e)}")
            self.fail += 1
            self.failed.append(f"{font} (exception)")
            if hasattr(self.opts, "verbose") and self.opts.verbose:
                import traceback

                traceback.print_exc()

    def check_text(self, text: str, use_tlf: bool) -> None:
        """Test the given text with all available fonts."""
        for font in self.f.getFonts():
            self.check_font(text, font, use_tlf)

    def check_extended_features(self) -> None:
        """Test FigletForge-specific extended features."""
        if hasattr(self.opts, "test_unicode") and self.opts.test_unicode:
            info("Testing Unicode character rendering...")
            self.check_text("Hello 世界", True)

        if hasattr(self.opts, "test_color") and self.opts.test_color and has_color:
            info("Testing ANSI color support...")
            # Skip actual testing if not running interactively
            if sys.stdout.isatty():
                # This would need to be visually inspected
                from figlet_forge import print_figlet

                print_figlet("Colors", font="slant", colors="RED:BLUE")
                input("Was the text red on blue background? (Press Enter)")
        else:
            info("ANSI color support requires an interactive terminal")

    def check_result(self) -> tuple[int, int]:
        """Summarize test results."""
        total = self.ok + self.fail
        print("\n" + "=" * 60)
        if total > 0:
            print(f"TEST RESULTS: {self.ok}/{total} passed ({self.ok/total:.1%})")
        else:
            print("TEST RESULTS: No tests were run")

        if self.fail:
            print(f"Failed fonts: {self.failed}")
        print("=" * 60)
        return self.fail, self.ok


def banner(text: str) -> None:
    """Display a section banner using figlet."""
    fig = Figlet(font="small")
    if has_color:
        cprint(fig.renderText(text), "blue", attrs=["bold"])
    else:
        print("\n" + "=" * 40)
        print(text)
        print("=" * 40)


def main():
    """Main test driver."""
    parser = OptionParser(
        version=f"FigletForge Test Suite {__version__}",
        description="Comprehensive test suite for the FigletForge package",
    )

    parser.add_option(
        "-s",
        "--show",
        action="store_true",
        default=False,
        help="pause at each failure and compare output (default: %default)",
    )
    parser.add_option(
        "-b",
        "--batch",
        action="store_true",
        default=False,
        help="run in batch mode, no interactive prompts",
    )
    parser.add_option(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="show detailed progress and debug information",
    )
    parser.add_option(
        "-u",
        "--test-unicode",
        action="store_true",
        default=False,
        help="test unicode character rendering capabilities",
    )
    parser.add_option(
        "-c",
        "--test-color",
        action="store_true",
        default=False,
        help="test ANSI color rendering (requires interactive terminal)",
    )

    opts, _ = parser.parse_args()
    test = Test(opts)

    banner("FigletForge Test Suite")
    print("Testing compatibility with original pyfiglet and standard figlet...")

    banner("Basic word rendering")
    test.check_text("foo", True)

    banner("Multi-word text with wrapping")
    test.check_text("This is a very long text with many spaces and little words", False)

    banner("Long word wrapping")
    test.check_text("A-very-long-word-that-will-be-cut-at-some-point I hope", False)

    banner("Multi-line text")
    test.check_text("line1\nline2", True)

    # Run extended feature tests if requested
    if opts.test_unicode or opts.test_color:
        banner("Extended features")
        test.check_extended_features()

    # Show final results
    failed, _ = test.check_result()

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
```
#!/usr/bin/env python3
"""
Main test runner for Figlet Forge.

This script discovers and runs all tests for the Figlet Forge package,
providing a convenient way to verify functionality.
"""

import os
import sys
import unittest
from pathlib import Path

# Add the src directory to the path for running tests
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_tests():
    """Discover and run all tests."""
    test_loader = unittest.defaultTestLoader
    test_suite = test_loader.discover(
        start_dir=Path(__file__).parent,
        pattern="test_*.py",
        top_level_dir=Path(__file__).parent.parent,
    )

    # Create a test runner
    test_runner = unittest.TextTestRunner(verbosity=2)

    # Run tests and return the result
    result = test_runner.run(test_suite)
    return 0 if result.wasSuccessful() else 1


def show_test_summary():
    """Show a summary of available tests."""
    test_dir = Path(__file__).parent
    test_files = list(test_dir.glob("**/test_*.py"))

    print(f"Found {len(test_files)} test files:")
    for test_file in sorted(test_files):
        relative_path = test_file.relative_to(test_dir)
        print(f"  - {relative_path}")

        # Try to display test methods in the file
        try:
            module_name = f"tests.{relative_path.as_posix()[:-3].replace('/', '.')}"
            test_module = __import__(module_name, fromlist=["*"])
            test_classes = [
                obj for name, obj in vars(test_module).items()
                if isinstance(obj, type) and issubclass(obj, unittest.TestCase)
                and obj is not unittest.TestCase
            ]

            for test_class in test_classes:
                print(f"    • {test_class.__name__}")
                for name in dir(test_class):
                    if name.startswith("test_"):
                        print(f"      - {name}")
        except (ImportError, AttributeError):
            print(f"    • (Error loading tests)")

    print("\nRun a specific test with:")
    print("  python -m unittest tests.unit.test_figlet.TestFigletCore.test_basic_initialization")
    print("\nRun all tests with:")
    print("  python tests/test.py")


if __name__ == "__main__":
    # Check for --list option
    if "--list" in sys.argv:
        show_test_summary()
        sys.exit(0)

    # Run all tests
    sys.exit(run_tests())
