#!/usr/bin/env python3

"""
Compatibility checker for Figlet Forge.

This script compares the output of figlet_forge with the output of the
standard figlet command line tool to verify compatibility.
"""

import subprocess
import sys
from pathlib import Path

# Add the source directory to path for imports
src_path = Path(__file__).parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

try:
    from figlet_forge.figlet import Figlet
except ImportError as e:
    print(f"Error importing Figlet: {e}")
    sys.exit(1)


def run_figlet(text, font="standard"):
    """Run the standard figlet command and return its output."""
    try:
        cmd = ["figlet", "-f", font, text]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running figlet: {e}")
        return f"ERROR: {e.stderr}"
    except FileNotFoundError:
        return "ERROR: figlet command not found"


def run_figlet_forge(text, font="standard"):
    """Run figlet_forge and return its output."""
    try:
        fig = Figlet(font=font)
        return fig.renderText(text)
    except Exception as e:
        import traceback

        traceback.print_exc()
        return f"ERROR: {str(e)}"


def compare_outputs(text, font="standard"):
    """Compare the outputs of figlet and figlet_forge."""
    print(f"=== COMPARING OUTPUTS for '{text}' with font '{font}' ===")

    figlet_output = run_figlet(text, font)
    forge_output = run_figlet_forge(text, font)

    print("\nSTANDARD FIGLET:")
    print("-" * 40)
    print(figlet_output)
    print("-" * 40)

    print("\nFIGLET FORGE:")
    print("-" * 40)
    print(forge_output)
    print("-" * 40)

    # Compare the outputs
    if figlet_output == forge_output:
        print("\n✅ MATCH: Outputs are identical!")
        return True
    else:
        print("\n❌ MISMATCH: Outputs differ!")

        # Detailed comparison for debugging
        figlet_lines = figlet_output.splitlines()
        forge_lines = forge_output.splitlines()

        max_lines = max(len(figlet_lines), len(forge_lines))

        print("\nDETAILED COMPARISON:")
        for i in range(max_lines):
            figlet_line = figlet_lines[i] if i < len(figlet_lines) else ""
            forge_line = forge_lines[i] if i < len(forge_lines) else ""

            if figlet_line == forge_line:
                status = "✓"
            else:
                status = "✗"

            print(f"{status} Line {i+1}:")
            print(f"  FIGLET: '{figlet_line}'")
            print(f"   FORGE: '{forge_line}'")

        return False


def main():
    """Run the compatibility check."""
    # Test cases to compare
    test_cases = [
        ("Hello", "standard"),
        ("World", "standard"),
        ("Eidosian", "standard"),
        ("Forge", "slant"),
    ]

    results = []

    for text, font in test_cases:
        result = compare_outputs(text, font)
        results.append(result)
        print("\n" + "=" * 60 + "\n")

    # Summary
    success = sum(1 for r in results if r)
    total = len(results)

    print(f"=== SUMMARY: {success}/{total} tests passed ===")

    return 0 if success == total else 1


if __name__ == "__main__":
    sys.exit(main())
