#!/usr/bin/env python3

"""
╔═════════════════════════════════════════════════════════════╗
║                                                             ║
║   ███████╗██╗ ██████╗ ██╗     ███████╗████████╗             ║
║   ██╔════╝██║██╔════╝ ██║     ██╔════╝╚══██╔══╝             ║
║   █████╗  ██║██║  ███╗██║     █████╗     ██║                ║
║   ██╔══╝  ██║██║   ██║██║     ██╔══╝     ██║                ║
║   ██║     ██║╚██████╔╝███████╗███████╗   ██║                ║
║   ╚═╝     ╚═╝ ╚═════╝ ╚══════╝╚══════╝   ╚═╝                ║
║                                                             ║
║   ███████╗ ██████╗ ██████╗  ██████╗ ███████╗               ║
║   ██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝               ║
║   █████╗  ██║   ██║██████╔╝██║  ███╗█████╗                 ║
║   ██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝                 ║
║   ██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗               ║
║   ╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝               ║
║                                                             ║
║   THE EIDOSIAN TYPOGRAPHIC ENGINE                           ║
║                                                             ║
╚═════════════════════════════════════════════════════════════╝

FIGLET FORGE: An Eidosian reimplementation extending pyfiglet with
colorized ANSI support, Unicode rendering and intelligent fallbacks
while maintaining backward compatibility.

Form follows function; elegance emerges from precision.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from figlet_forge import Figlet, print_figlet
    from figlet_forge.color.effects import gradient_colorize, rainbow_colorize
    from figlet_forge.version import __version__
except ImportError as e:
    print(f"Error: Could not import Figlet Forge modules: {e}")
    print("Make sure the package is installed or the 'src' directory is accessible.")
    sys.exit(1)


def display_eidosian_manifesto():
    """Display the Figlet Forge manifesto with Eidosian styling."""
    # Get terminal width
    try:
        term_width = os.get_terminal_size().columns
    except (AttributeError, OSError):
        term_width = 80

    # Print header
    print("\n" + "═" * term_width)
    print(f"⚛️  FIGLET FORGE v{__version__} - THE EIDOSIAN TYPOGRAPHIC ENGINE ⚡")
    print("═" * term_width)

    # Render main title with gradient
    fig = Figlet(font="slant", width=term_width, justify="center")
    title = fig.renderText("FIGLET FORGE")
    colored_title = gradient_colorize(title, "CYAN", "MAGENTA")
    print(colored_title)

    # Core principles
    principles = [
        "Contextual Integrity",
        "Exhaustive But Concise",
        "Flow Like Water, Strike Like Lightning",
        "Recursive Refinement",
        "Precision as Style",
    ]

    small_fig = Figlet(font="small", width=term_width, justify="center")
    print("\n" + "─" * term_width)
    subtitle = small_fig.renderText("EIDOSIAN PRINCIPLES")
    print(rainbow_colorize(subtitle))
    print("─" * term_width + "\n")

    # Display principles
    for i, principle in enumerate(principles):
        print(f"  {i+1}. \033[1;36m{principle}\033[0m")

    print("\n" + "─" * term_width)

    # Show a sample of different fonts
    fonts = ["standard", "slant", "small", "big", "mini"]
    sample_text = "Hello World"

    print("\n\033[1mFONT SAMPLES:\033[0m\n")
    for font in fonts:
        try:
            fig = Figlet(font=font)
            result = fig.renderText(sample_text)
            print(f"\033[1;33m{font}:\033[0m")
            print(result)
            print()
        except Exception as e:
            print(f"Error rendering font {font}: {e}")

    # Show color capabilities
    print("\n\033[1mCOLOR CAPABILITIES:\033[0m\n")

    fig = Figlet(font="small", width=term_width)
    color_text = fig.renderText("Color Magic")

    print("\033[1;33mRainbow colors:\033[0m")
    print(rainbow_colorize(color_text))
    print("\n\033[1;33mGradient colors:\033[0m")
    print(gradient_colorize(color_text, "RED", "BLUE"))

    # Footer
    print("\n" + "═" * term_width)
    quote = '"Form follows function; elegance emerges from precision."'
    print(f"\033[3m{quote}\033[0m".center(term_width))
    print("═" * term_width + "\n")


if __name__ == "__main__":
    display_eidosian_manifesto()
