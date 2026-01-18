#!/usr/bin/env python
"""
Command line interface for Figlet Forge.

This module provides the CLI entry point and command processing
functionality for the Figlet Forge package.
"""
import argparse
import sys
import textwrap
from typing import List, Optional

from ..color import get_coloring_functions
from ..core.exceptions import FigletError, FontNotFound
from ..core.utils import get_terminal_size
from ..figlet import Figlet
from ..version import __version__
from .showcase import ColorShowcase, display_color_showcase, generate_showcase

# Default values
DEFAULT_WIDTH = 80


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command line arguments.

    Args:
        args: Command line arguments (defaults to sys.argv[1:])

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Figlet Forge - ASCII art text generator with advanced features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              figlet_forge "Hello World"
              figlet_forge --font=slant "Hello World"
              figlet_forge --color=rainbow "Hello World"
              figlet_forge --color=red:blue --border=double "Hello World"
              figlet_forge --flip --reverse "Hello World"
              figlet_forge --version
            """
        ),
    )

    text_input = parser.add_argument_group("Text Input")
    text_input.add_argument(
        "text", nargs="*", help="Text to convert (reads from STDIN if not provided)"
    )

    font_options = parser.add_argument_group("Font Options")
    font_options.add_argument("--font", "-f", help="Font to use (default: standard)")
    font_options.add_argument(
        "--list-fonts", "-l", action="store_true", help="List available fonts"
    )

    layout_options = parser.add_argument_group("Layout Options")
    layout_options.add_argument(
        "--width",
        "-w",
        type=int,
        help="Width of output (default: terminal width or 80)",
    )
    layout_options.add_argument(
        "--justify",
        "-j",
        choices=["left", "right", "center", "auto"],
        help="Text justification (default: auto)",
    )
    layout_options.add_argument(
        "--direction",
        "-d",
        choices=["auto", "left-to-right", "right-to-left"],
        help="Text direction (default: auto)",
    )

    transform_options = parser.add_argument_group("Transformation Options")
    transform_options.add_argument(
        "--reverse", "-r", action="store_true", help="Reverse the text direction"
    )
    transform_options.add_argument(
        "--flip", "-F", action="store_true", help="Flip the text vertically"
    )
    transform_options.add_argument(
        "--border",
        choices=["single", "double", "rounded", "bold", "shadow", "ascii"],
        help="Add border around the text",
    )
    transform_options.add_argument(
        "--shade", action="store_true", help="Add shading/shadow effect"
    )

    color_options = parser.add_argument_group("Color Options")
    # Make color option accept a value but also work as a flag
    color_options.add_argument(
        "--color",
        "-c",
        nargs="?",
        const="RAINBOW",
        help="Color specification (NAME, NAME:BG, rgb;g;b, or rainbow/gradient)",
    )
    color_options.add_argument(
        "--color-list",
        action="store_true",
        help="List available colors",
    )

    output_options = parser.add_argument_group("Output Options")
    output_options.add_argument(
        "--unicode", "-u", action="store_true", help="Enable Unicode character support"
    )
    output_options.add_argument(
        "--output", "-o", help="File to write output to (default: STDOUT)"
    )
    output_options.add_argument("--html", action="store_true", help="Output as HTML")
    output_options.add_argument("--svg", action="store_true", help="Output as SVG")

    showcase_options = parser.add_argument_group("Showcase Options")
    showcase_options.add_argument(
        "--showcase",
        "--sample",
        dest="showcase",
        action="store_true",
        help="Show fonts and styles showcase",
    )
    showcase_options.add_argument(
        "--sample-text",
        default="hello",
        help="Text to use in showcase",
    )
    showcase_options.add_argument(
        "--sample-color",
        nargs="?",
        const="ALL",
        help="Color to use in showcase or 'ALL' for color showcase",
    )
    showcase_options.add_argument(
        "--sample-fonts",
        nargs="?",
        const="ALL",
        help="Comma-separated list of fonts to include in showcase or 'ALL' for all fonts",
    )

    # Add version option
    parser.add_argument(
        "--version", "-v", action="store_true", help="Show version information"
    )

    parsed_args = parser.parse_args(args)
    return parsed_args


def read_input() -> str:
    """Read input text from stdin if available, with proper error handling."""
    if sys.stdin.isatty():
        return ""

    try:
        text = sys.stdin.read()
        return text.rstrip()
    except KeyboardInterrupt:
        return ""
    except Exception as e:
        print(f"Error reading from stdin: {e}", file=sys.stderr)
        return ""


def list_colors() -> None:
    """Display a list of available colors."""
    categories = ColorShowcase.get_color_categories()

    print("Available colors:")
    print("---------------")

    # Print colors by category
    for category, colors in categories.items():
        print(f"\n{category} colors:")
        for color in colors:
            try:
                # Import here to avoid circular imports
                from ..color import colored_format

                # Generate a colored sample
                sample = colored_format("███████", color=color)
                print(f"  {color.ljust(15)} {sample}")
            except Exception:
                print(f"  {color}")

    print("\nColor formats:")
    print("  NAME            - e.g., RED, BLUE, GREEN")
    print("  NAME:NAME       - e.g., WHITE:BLUE (foreground:background)")
    print("  N;N;N          - e.g., 255;0;0 (RGB values)")
    print("  gradient_name   - e.g., red_to_blue, yellow_to_green")


def main(argv: Optional[List[str]] = None) -> int:
    """
    Main entry point for the Figlet Forge CLI.

    Args:
        argv: Command line arguments (defaults to sys.argv[1:])

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    try:
        args = parse_args(argv)

        # Show version if requested
        if args.version:
            print(f"Figlet Forge v{__version__}")
            return 0

        # List fonts if requested
        if args.list_fonts:
            figlet = Figlet()
            fonts = figlet.get_fonts()
            print(f"Available fonts ({len(fonts)}):")
            for i, font in enumerate(sorted(fonts)):
                print(f"{font.ljust(20)}", end=" " if (i + 1) % 4 != 0 else "\n")
            if len(fonts) % 4 != 0:
                print()  # Final newline if needed
            return 0

        # List colors if requested
        if args.color_list:
            list_colors()
            return 0

        # Determine width
        width = args.width
        if not width:
            term_width, _ = get_terminal_size()
            width = term_width if term_width else DEFAULT_WIDTH

        # Show color showcase
        if args.sample_color and args.sample_color.upper() == "ALL":
            display_color_showcase(
                sample_text=args.sample_text, font=args.font or "small"
            )
            return 0

        # Show regular showcase if requested
        if args.showcase:
            showcase = generate_showcase(
                sample_text=args.sample_text,
                fonts=args.sample_fonts,
                color=args.sample_color,
                width=width,
            )
            print(showcase)

            # Show usage guide
            print("\n" + "=" * width)
            print(" ⚛️ FIGLET FORGE USAGE GUIDE - THE EIDOSIAN WAY ⚡".center(width))
            print("=" * width + "\n")

            # Usage guide content
            guide = [
                "📊 SHOWCASE SUMMARY:",
                "  • Displayed fonts and color styles",
                "  • Use the commands below to apply these styles to your own text",
                "",
                "📋 CORE USAGE PATTERNS:",
                "  figlet_forge 'Your Text Here'              # Simple rendering",
                "  cat file.txt | figlet_forge                # Pipe text from stdin",
                "  figlet_forge 'Line 1\\nLine 2'              # Multi-line text",
                "",
                "🔤 FONT METAMORPHOSIS:",
                "  Command format: figlet_forge --font=<font_name> 'Your Text'",
                "",
                "  Available fonts you've just seen:",
                "",
                "    Display Fonts:",
                "      figlet_forge --font=banner 'Your Text'  # Wide, horizontally stretched text for announcem...",
                "      figlet_forge --font=big 'Your Text'  # Bold, attention-grabbing display for headlines",
                "      figlet_forge --font=block 'Your Text'  # Solid, impactful lettering for emphasis",
                "      figlet_forge --font=shadow 'Your Text'  # Dimensional text with drop shadows for depth",
                "      figlet_forge --font=standard 'Your Text'  # The classic figlet font, perfect for general use",
                "",
                "    Stylized Fonts:",
                "      figlet_forge --font=bubble 'Your Text'  # Rounded, friendly letters for approachable mess...",
                "      figlet_forge --font=ivrit 'Your Text'  # Right-to-left oriented font for Hebrew-style text",
                "      figlet_forge --font=lean 'Your Text'  # Condensed characters for fitting more text hori...",
                "",
                "    Technical Fonts:",
                "      figlet_forge --font=digital 'Your Text'  # Technical, LCD-like display for a technological...",
                "",
                "    Compact Fonts:",
                "      figlet_forge --font=mini 'Your Text'  # Ultra-compact font for constrained spaces",
                "      figlet_forge --font=small 'Your Text'  # Compact representation when space is limited",
                "      figlet_forge --font=smslant 'Your Text'  # Small slanted font for compact, dynamic text",
                "",
                "    Script Fonts:",
                "      figlet_forge --font=script 'Your Text'  # Elegant, cursive-like appearance for a sophisti...",
                "      figlet_forge --font=slant 'Your Text'  # Adds a dynamic, forward-leaning style to text",
                "      figlet_forge --font=smscript 'Your Text'  # Small script font for elegant, compact text",
                "",
                "📐 SPATIAL ARCHITECTURE:",
                "  --width=120                  # Set output width",
                "  --justify=center             # Center-align text (left, right, center)",
                "  --direction=right-to-left    # Change text direction",
                "",
                "🔄 RECURSIVE TRANSFORMATIONS:",
                "  --reverse                    # Mirror text horizontally",
                "  --flip                       # Flip text vertically (upside-down)",
                "  --border=single              # Add border (single, double, rounded, bold, ascii)",
                "  --shade                      # Add shadow effect",
                "",
                "🎨 COLOR EFFECTS:",
                "  --color=RED                  # Basic color",
                "  --color=RED:BLACK            # Foreground:Background color",
                "  --color=rainbow              # Rainbow effect",
                "  --color=red_to_blue          # Gradient effect",
                "  --color=green_on_black       # Preset color style",
                "  --color-list                 # Show available colors",
                "",
                "✨ SYNTHESIS PATTERNS:",
                "  # Rainbow colored slant font with border",
                "  figlet_forge --font=slant --color=rainbow --border=single 'Synthesis'",
                "",
                "  # Center-justified big green text on black background",
                "  figlet_forge --font=big --color=green_on_black --justify=center 'Elegant'",
                "",
                "  # Flipped and reversed text with shadow effect",
                "  figlet_forge --font=standard --reverse --flip --shade 'Recursion'",
                "",
                "📋 COMMAND LINE OPTIONS:",
                "  --showcase                   # Show fonts and styles showcase",
                "  --sample                     # Equivalent to --showcase",
                '  --sample-text="Hello World"  # Set sample text for showcase',
                "  --sample-color=RED           # Set color for samples",
                "  --sample-fonts=slant,mini    # Specify which fonts to sample",
                "  --unicode, -u                # Enable Unicode character support",
                "  --version, -v                # Display version information",
                "  --help                       # Show help message",
                "",
                "=" * width,
                " 🚀 WHAT'S NEXT?".center(width),
                "=" * width,
                "",
                "1. Try creating your own ASCII art:",
                "   figlet_forge --font=slant --color=rainbow 'Your Custom Text'",
                "",
                "2. Explore more options:",
                "   figlet_forge --help",
                "",
                "3. Save your creations:",
                "   figlet_forge --font=big --color=blue_on_black 'Hello' --output=banner.txt",
                "",
                "4. Combine with other tools:",
                "   figlet_forge 'Welcome' | lolcat",
                "   echo 'Hello' | figlet_forge --font=mini --border=rounded",
                "",
                "5. Create a login banner:",
                "   figlet_forge --font=big --color=green_on_black --border=double 'SYSTEM LOGIN' > /etc/motd",
                "",
                f"⚛️ Figlet Forge v{__version__} ⚡ - Eidosian Typography Engine",
                '  "Form follows function; elegance emerges from precision."',
            ]

            for line in guide:
                print(line)

            return 0

        # Get input text for regular rendering mode
        text = " ".join(args.text) if args.text else read_input()
        if not text:
            print("No input provided. Use 'figlet_forge --help' for usage information.")
            return 0

        # Create Figlet instance
        figlet = Figlet(
            font=args.font, width=width, justify=args.justify, direction=args.direction
        )

        # Render text
        result = figlet.renderText(text)

        # Apply transformations in sequence
        if args.reverse:
            result = result.reverse()
        if args.flip:
            result = result.flip()
        if args.shade:
            result = result.shadow()
        elif args.border:
            result = result.border(style=args.border)

        # Apply color if specified
        if args.color:
            color_func = get_coloring_functions(args.color)
            if color_func:
                result = color_func(result)

        # Handle output formatting
        if args.html:
            from ..render.figlet_engine import RenderEngine

            result = RenderEngine.to_html(result)
        elif args.svg:
            from ..render.figlet_engine import RenderEngine

            result = RenderEngine.to_svg(result)

        # Output to file or stdout
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(str(result))
        else:
            print(result)

        return 0

    except FontNotFound as e:
        print(f"Font not found: {e}", file=sys.stderr)
        return 1
    except FigletError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nOperation canceled by user.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
