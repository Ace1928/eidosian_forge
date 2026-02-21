from eidosian_core import eidosian
#!/usr/bin/env python
"""
Command line interface for Figlet Forge.

This module provides the CLI entry point and command processing
functionality for the Figlet Forge package.
"""
import argparse
import re
import sys
import textwrap
from typing import List, Optional

from ..color.figlet_color import COLOR_CODES, colored_format, get_coloring_functions
from ..color import color_formats
from ..core.exceptions import FigletError, FontNotFound
from ..core.utils import get_terminal_size
from ..figlet import Figlet
from ..version import __version__
from .showcase import ColorShowcase, display_color_showcase, generate_showcase

# Default values
DEFAULT_WIDTH = 80


@eidosian()
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
    font_options.add_argument(
        "--font", "-f", default="standard", help="Font to use (default: standard)"
    )
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
        nargs="?",
        const=True,  # When flag is used without value
        default=False,  # When flag is not present
        help="Show fonts and styles showcase (optionally provide sample text)",
    )
    showcase_options.add_argument(
        "--sample-text",
        nargs="?",
        const="Hello Eidos",  # Default when flag used without value
        default="hello",  # Default when flag not used at all
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

    effective_args = list(args) if args is not None else None
    if effective_args is not None:
        known_color_tokens = {k.lower() for k in COLOR_CODES.keys()} | {
            "rainbow",
            "random",
            "all",
            "list",
            "red_to_blue",
            "yellow_to_green",
            "magenta_to_cyan",
            "white_to_blue",
            "green_on_black",
            "red_on_black",
            "yellow_on_blue",
            "white_on_red",
            "black_on_white",
            "cyan_on_black",
        }

        def _looks_like_color(token: str) -> bool:
            t = token.strip()
            tl = t.lower()
            if tl in known_color_tokens:
                return True
            if ":" in t or "_to_" in tl or "_on_" in tl:
                return True
            return re.fullmatch(r"\d{1,3};\d{1,3};\d{1,3}", t) is not None

        normalized: List[str] = []
        i = 0
        while i < len(effective_args):
            tok = effective_args[i]
            if tok == "--color":
                # Preserve "--color" flag semantics even when followed by text.
                nxt = effective_args[i + 1] if i + 1 < len(effective_args) else None
                if nxt and not nxt.startswith("-"):
                    if _looks_like_color(nxt):
                        normalized.append(tok)
                        normalized.append(nxt)
                        i += 2
                        continue
                    normalized.append("--color=RAINBOW")
                    normalized.append(nxt)
                    i += 2
                    continue
            normalized.append(tok)
            i += 1
        effective_args = normalized

    parsed_args = parser.parse_args(effective_args)
    if isinstance(parsed_args.showcase, str):
        parsed_args.sample_text = parsed_args.showcase
        parsed_args.showcase = True
    return parsed_args


@eidosian()
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


@eidosian()
def list_colors() -> None:
    """Display a list of available colors."""
    print("Available color names:")
    print("---------------------")
    for color_name in sorted(COLOR_CODES.keys()):
        print(f"  {color_name}")
    print()
    print("\nSpecial color formats:")
    for name, desc in color_formats.items():
        print(f"  {name}: {desc}")
    print()

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
    print("\nUsage examples:")
    print("  figlet_forge --color=RED 'Hello'")
    print("  figlet_forge --color=rainbow 'Hello'")


@eidosian()
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
            print(f"Figlet Forge version {__version__}")
            return 0

        # List fonts if requested
        if args.list_fonts:
            figlet = Figlet()
            fonts = figlet.get_fonts()
            print("Available fonts:")
            print(f"Count: {len(fonts)}")
            for i, font in enumerate(sorted(fonts)):
                print(f"{font.ljust(20)}", end=" " if (i + 1) % 4 != 0 else "\n")
            if len(fonts) % 4 != 0:
                print()  # Final newline if needed
            return 0

        # List colors if requested
        if args.color_list or (args.color and args.color.lower() == "list"):
            list_colors()
            return 0

        # Determine width
        width = args.width
        if not width:
            term_width, _ = get_terminal_size()
            width = term_width if term_width else DEFAULT_WIDTH

        # Show color showcase if --sample-color=ALL is used (even without --showcase flag)
        if args.sample_color and args.sample_color.upper() == "ALL" and not args.showcase:
            display_color_showcase(
                sample_text=args.sample_text, font=args.font or "small"
            )
            return 0

        # Treat sample option family as implicit showcase mode for backward
        # compatibility with integration tests and legacy CLI usage.
        if not args.showcase:
            if args.sample_color is not None or args.sample_fonts is not None:
                args.showcase = True
            elif args.sample_text != "hello":
                args.showcase = True

        # Show regular showcase if requested (handles all showcase cases including color showcase)
        if args.showcase:
            try:
                # If showcase was given a value (e.g., --showcase=Text), use it as sample_text
                sample_text = args.showcase if isinstance(args.showcase, str) else args.sample_text
                
                # Parse fonts from comma-separated string if provided
                fonts = args.sample_fonts
                if fonts and fonts != "ALL" and isinstance(fonts, str) and "," in fonts:
                    fonts = [f.strip() for f in fonts.split(",")]
                
                showcase = generate_showcase(
                    sample_text=sample_text,
                    fonts=fonts,
                    color=args.sample_color,
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
                    "  --color-list                 # Show available colors",
                    "",
                    f"⚛️ Figlet Forge v{__version__} ⚡ - Eidosian Typography Engine",
                ]

                for line in guide:
                    print(line)

                return 0
            except Exception as e:
                print(f"Error generating showcase: {e}", file=sys.stderr)
                return 1

        # Get input text for regular rendering mode
        # Handle case where --color might have consumed what should be text
        # (e.g., `--color Hello` where Hello looks like a color value but isn't)
        text = " ".join(args.text) if args.text else read_input()
        
        # Check if color value looks like text (not a valid color specification)
        # Valid colors: NAME, NAME:NAME, rgb;g;b, rainbow, gradient specs
        if not text and args.color and args.color not in (None, "RAINBOW"):
            # Check if color is actually meant to be text
            color_upper = args.color.upper()
            known_colors = {
                "RED", "GREEN", "BLUE", "YELLOW", "CYAN", "MAGENTA", "WHITE", "BLACK",
                "LIGHT_RED", "LIGHT_GREEN", "LIGHT_BLUE", "LIGHT_YELLOW", 
                "LIGHT_CYAN", "LIGHT_MAGENTA", "LIGHT_WHITE", "LIGHT_BLACK",
                "RAINBOW", "RANDOM", "ALL"
            }
            is_rgb = ";" in args.color and all(p.isdigit() for p in args.color.split(";"))
            is_fg_bg = ":" in args.color
            is_gradient = "_TO_" in color_upper or "_ON_" in color_upper
            is_known = color_upper in known_colors
            
            if not (is_known or is_rgb or is_fg_bg or is_gradient):
                # Color value looks like text, treat it as such
                text = args.color
                args.color = "RAINBOW"  # Default color when using --color as flag
        
        if not text:
            print("No input provided. Use 'figlet_forge --help' for usage information.", file=sys.stderr)
            return 1  # Error: no input

        # Create Figlet instance
        figlet = Figlet(
            font=args.font,
            width=width,
            justify=args.justify,
            direction=args.direction,
            unicode_aware=args.unicode if hasattr(args, 'unicode') else False,
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
        if args.border:
            result = result.border(style=args.border)

        # Apply color if specified
        if args.color:
            color_name = str(args.color).strip().lower()
            coloring_funcs = get_coloring_functions()

            if color_name == "rainbow":
                if callable(coloring_funcs):
                    result = coloring_funcs(str(result))
                    color_func = None
                elif isinstance(coloring_funcs, dict):
                    color_func = coloring_funcs.get("rainbow")
                else:
                    color_func = None
                if color_func:
                    result = color_func(str(result))
                elif not callable(coloring_funcs):
                    print("Error: Rainbow color effect unavailable", file=sys.stderr)
                    return 1
            elif "_to_" in color_name:
                parts = color_name.split("_to_")
                if len(parts) == 2:
                    gradient_func = (
                        coloring_funcs.get("gradient")
                        if isinstance(coloring_funcs, dict)
                        else None
                    )
                    if not gradient_func:
                        print("Error: Gradient effect unavailable", file=sys.stderr)
                        return 1
                    result = gradient_func(str(result), parts[0], parts[1])
                else:
                    print(
                        f"Error: Invalid gradient specification '{color_name}'",
                        file=sys.stderr,
                    )
                    return 1
            else:
                colored = colored_format(str(result), color_name)
                if colored == str(result):
                    print(
                        f"Error: Invalid color specification '{color_name}'",
                        file=sys.stderr,
                    )
                    return 1
                result = colored

        # Handle output formatting
        if args.html:
            from ..render.figlet_engine import RenderEngine

            result = RenderEngine.to_html(str(result))
        elif args.svg:
            from ..render.figlet_engine import RenderEngine

            result = RenderEngine.to_svg(str(result))

        # Output to file or stdout
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(str(result))
                f.write("\n")  # Ensure trailing newline
        else:
            print(result)

        return 0

    except SystemExit as e:
        # argparse uses SystemExit for parse failures; return the exit code so
        # callers (including tests) can assert on error handling.
        code = e.code if isinstance(e.code, int) else 2
        return code
    except FontNotFound as e:
        # Get just the font name from the exception
        font_name = str(e).split(" - ")[0] if " - " in str(e) else str(e)
        print(f"Error: Font not found - {font_name}", file=sys.stderr)
        return 1
    except FigletError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2  # General exception return code


if __name__ == "__main__":
    sys.exit(main())
