from eidosian_core import eidosian
"""
Showcase module for Figlet Forge.

This module provides functionality for showcasing fonts and styles available
in Figlet Forge, allowing users to preview different options.
"""

import logging
from typing import List, Optional

from ..color.figlet_color import COLORS, parse_color
from ..core.exceptions import FigletError, FontNotFound
from ..figlet import Figlet
from ..version import PROGRAM_NAME, VERSION

# Configure logger for this module
logger = logging.getLogger(__name__)


@eidosian()
def show_font_showcase(
    sample_text: str = "hello",
    fonts: Optional[List[str]] = None,
    color: str = "",
    width: int = 80,
    justify: str = "auto",
) -> None:
    """
    Display a showcase of different fonts using the specified text.

    This function renders the sample text in various fonts to demonstrate
    the available options in Figlet Forge.

    Args:
        sample_text: Text to use for showcase
        fonts: List of fonts to showcase (default: selected popular fonts)
        color: Color for the output (e.g., "RED", "BLUE:BLACK")
        width: Maximum width for output
        justify: Text justification ('auto', 'left', 'center', 'right')
    """
    # Define a title bar with program branding
    title = f"⚛️ {PROGRAM_NAME} FONT SHOWCASE - {sample_text.upper()} ⚡"

    # Print header
    divider = "=" * 132
    print(divider)
    print(title.center(132))
    print(divider)
    print()

    # Default showcase fonts if none provided
    if fonts is None:
        # These are fonts that should be available in base installation
        fonts = ["standard", "slant", "small", "mini", "big"]

    print(f"Sample text: '{sample_text}'")
    print(f"Showcasing {len(fonts)} fonts")
    print()

    try:
        # Parse color specifications once
        fg_color, bg_color = "", ""
        if color:
            try:
                fg_color, bg_color = parse_color(color)
            except Exception as e:
                print(f"Warning: Invalid color specification - {e}")
                color = ""
    except Exception as e:
        print(f"Error initializing Figlet: {e}")
        return

    # Keep track of fonts we've successfully displayed
    displayed_fonts = 0

    # Show each font
    for font_name in fonts:
        try:
            # Section divider for each font
            print("=" * 50)
            print(f" FONT: {font_name}")
            print("=" * 50)
            print()

            # Attempt to load the font with clear feedback
            print(f"Attempting to load font: {font_name}...")
            fig = Figlet(font=font_name, width=width, justify=justify)

            # Store the actual font name that was loaded (might be fallback)
            actual_font_name = fig.font

            # Verify the font was loaded and it's a proper FigletFont instance
            if not hasattr(fig, "Font") or fig.Font is None:
                print(f"✗ Failed to load font '{font_name}', missing Font instance")
                continue

            # Show where the font was loaded from
            loaded_from = getattr(fig.Font, "loaded_from", "unknown location")
            print(f"✓ Loaded font '{actual_font_name}' from {loaded_from}")

            print(f"✓ Font '{actual_font_name}' loaded successfully")

            try:
                # Render the sample text with this font
                rendered = fig.render_text(sample_text)

                # Apply color if specified
                if color and fg_color:
                    # Apply colors directly to the rendered output
                    final_output = fg_color + bg_color + str(rendered) + "\033[0m"
                    print(final_output)
                else:
                    print(rendered)

                displayed_fonts += 1
            except Exception as e:
                print(f"Error rendering with font '{actual_font_name}': {e}")

        except FontNotFound as e:
            print(f"✗ Font '{font_name}' not found: {e}")
        except FigletError as e:
            print(f"✗ Error with font '{font_name}': {e}")
        except Exception as e:
            print(f"✗ Unexpected error with font '{font_name}': {e}")

        print()

    # Print footer message
    print(divider)
    print(" END OF SHOWCASE".center(132))
    print(divider)
    print()


@eidosian()
def show_color_showcase(
    sample_text: str = "hello",
    font: str = "small",
    width: int = 80,
    justify: str = "auto",
) -> None:
    """
    Display a showcase of different color options.

    This function demonstrates the various colors and color effects
    available in Figlet Forge.

    Args:
        sample_text: Text to use for showcase
        font: Font to use for showcase
        width: Maximum width for output
        justify: Text justification ('auto', 'left', 'center', 'right')
    """
    # Load the font for reuse across all color examples
    try:
        fig = Figlet(font=font, width=width, justify=justify)
        rendered = fig.render_text(sample_text)
    except Exception as e:
        print(f"Error initializing showcase: {e}")
        return

    # Define a title bar with program branding
    title = f"⚛️ {PROGRAM_NAME} COLOR SHOWCASE ⚡"

    # Print header
    divider = "=" * 80
    print(divider)
    print(title.center(80))
    print(divider)
    print()

    print(f"Sample text: '{sample_text}'")
    print(f"Using font: '{fig.font}'")
    print()
    print()

    # Showcase basic colors
    print("Basic colors:")
    print("-------------")
    print()
    _show_colors(rendered, COLORS[:8])

    # Showcase bright colors
    print("Bright colors:")
    print("--------------")
    print()
    _show_colors(rendered, [f"LIGHT_{c}" for c in COLORS[:8]])

    # Showcase effects
    print("Effects colors:")
    print("---------------")
    print()
    effects = [
        "RAINBOW",
        "RED_TO_BLUE",
        "YELLOW_TO_GREEN",
        "MAGENTA_TO_CYAN",
        "WHITE_TO_BLUE",
        "BLACK_ON_WHITE",
        "WHITE_ON_BLACK",
    ]
    _show_colors(rendered, effects)

    # Show some gradient examples with custom colors
    print("Gradient Examples:")
    print("-----------------")
    print()
    gradient_pairs = [
        "RED to BLUE",
        "YELLOW to GREEN",
        "CYAN to MAGENTA",
        "WHITE to BLACK",
    ]
    _show_gradients(rendered, gradient_pairs)

    # Print footer
    print(divider)
    print("END OF COLOR SHOWCASE".center(80))
    print(divider)


def _show_colors(rendered: str, colors: List[str]) -> None:
    """
    Show rendered text in specified colors.

    Args:
        rendered: Pre-rendered text to colorize
        colors: List of color names to apply
    """
    for color_name in colors:
        try:
            # Add padding to ensure consistent layout
            print(f"{color_name}:")

            # Parse the color specification
            fg_color, bg_color = parse_color(color_name)

            # Apply colors to the rendered text
            if fg_color or bg_color:
                colored_text = fg_color + bg_color + str(rendered) + "\033[0m"
                print(colored_text)
            else:
                print(rendered)

        except Exception as e:
            print(f"Error with color {color_name}: {e}")
        print()


def _show_gradients(rendered: str, gradient_pairs: List[str]) -> None:
    """
    Show rendered text with gradient effects.

    Args:
        rendered: Pre-rendered text to apply gradients to
        gradient_pairs: List of color pairs for gradients (e.g., "RED to BLUE")
    """
    for gradient in gradient_pairs:
        print(f"{gradient}:")
        try:
            # Extract the two colors from the gradient description
            color1, color2 = gradient.split(" to ")

            # Create a custom gradient effect
            gradient_spec = f"{color1.strip()}_{color2.strip()}"
            fg_color, bg_color = parse_color(gradient_spec)

            # Apply colors to the rendered text
            if fg_color:
                colored_text = fg_color + (bg_color or "") + str(rendered) + "\033[0m"
                print(colored_text)
            else:
                # Fallback if gradient not supported
                print(rendered)

        except Exception as e:
            print(f"Error with gradient {gradient}: {e}")
        print()


@eidosian()
def show_usage_guide() -> None:
    """Display an Eidosian-style usage guide with examples and tips."""
    divider = "=" * 132
    title = "⚛️ FIGLET FORGE USAGE GUIDE - THE EIDOSIAN WAY ⚡"

    # Print header
    print(divider)
    print(title.center(132))
    print(divider)
    print()

    # Summary section
    print("📊 SHOWCASE SUMMARY:")
    print("  • Displayed fonts and color styles")
    print("  • Use the commands below to apply these styles to your own text")
    print()

    # Core usage
    print("📋 CORE USAGE PATTERNS:")
    print("  figlet_forge 'Your Text Here'              # Simple rendering")
    print("  cat file.txt | figlet_forge                # Pipe text from stdin")
    print("  figlet_forge 'Line 1\\nLine 2'              # Multi-line text")
    print()

    # Font section
    print("🔤 FONT METAMORPHOSIS:")
    print("  Command format: figlet_forge --font=<font_name> 'Your Text'")
    print()
    print("  Available fonts you've just seen:")
    print()
    print("    Display Fonts:")
    print(
        "      figlet_forge --font=banner 'Your Text'  # Wide, horizontally stretched text for announcem..."
    )
    print(
        "      figlet_forge --font=big 'Your Text'  # Bold, attention-grabbing display for headlines"
    )
    print(
        "      figlet_forge --font=block 'Your Text'  # Solid, impactful lettering for emphasis"
    )
    print(
        "      figlet_forge --font=shadow 'Your Text'  # Dimensional text with drop shadows for depth"
    )
    print(
        "      figlet_forge --font=standard 'Your Text'  # The classic figlet font, perfect for general use"
    )
    print()
    print("    Stylized Fonts:")
    print(
        "      figlet_forge --font=bubble 'Your Text'  # Rounded, friendly letters for approachable mess..."
    )
    print(
        "      figlet_forge --font=ivrit 'Your Text'  # Right-to-left oriented font for Hebrew-style text"
    )
    print(
        "      figlet_forge --font=lean 'Your Text'  # Condensed characters for fitting more text hori..."
    )
    print()
    print("    Technical Fonts:")
    print(
        "      figlet_forge --font=digital 'Your Text'  # Technical, LCD-like display for a technological..."
    )
    print()
    print("    Compact Fonts:")
    print(
        "      figlet_forge --font=mini 'Your Text'  # Ultra-compact font for constrained spaces"
    )
    print(
        "      figlet_forge --font=small 'Your Text'  # Compact representation when space is limited"
    )
    print(
        "      figlet_forge --font=smslant 'Your Text'  # Small slanted font for compact, dynamic text"
    )
    print()
    print("    Script Fonts:")
    print(
        "      figlet_forge --font=script 'Your Text'  # Elegant, cursive-like appearance for a sophisti..."
    )
    print(
        "      figlet_forge --font=slant 'Your Text'  # Adds a dynamic, forward-leaning style to text"
    )
    print(
        "      figlet_forge --font=smscript 'Your Text'  # Small script font for elegant, compact text"
    )
    print()

    # Layout options
    print("📐 SPATIAL ARCHITECTURE:")
    print("  --width=120                  # Set output width")
    print("  --justify=center             # Center-align text (left, right, center)")
    print("  --direction=right-to-left    # Change text direction")
    print()

    # Transformations
    print("🔄 RECURSIVE TRANSFORMATIONS:")
    print("  --reverse                    # Mirror text horizontally")
    print("  --flip                       # Flip text vertically (upside-down)")
    print(
        "  --border=single              # Add border (single, double, rounded, bold, ascii)"
    )
    print("  --shade                      # Add shadow effect")
    print()

    # Color options
    print("🎨 COLOR EFFECTS:")
    print("  --color=RED                  # Basic color")
    print("  --color=RED:BLACK            # Foreground:Background color")
    print("  --color=rainbow              # Rainbow effect")
    print("  --color=red_to_blue          # Gradient effect")
    print("  --color=green_on_black       # Preset color style")
    print("  --color-list                 # Show available colors")
    print()

    # Combined examples
    print("✨ SYNTHESIS PATTERNS:")
    print("  # Rainbow colored slant font with border")
    print("  figlet_forge --font=slant --color=rainbow --border=single 'Synthesis'")
    print()
    print("  # Center-justified big green text on black background")
    print("  figlet_forge --font=big --color=green_on_black --justify=center 'Elegant'")
    print()
    print("  # Flipped and reversed text with shadow effect")
    print("  figlet_forge --font=standard --reverse --flip --shade 'Recursion'")
    print()

    # Command line reference
    print("📋 COMMAND LINE OPTIONS:")
    print("  --showcase                   # Show fonts and styles showcase")
    print("  --sample                     # Equivalent to --showcase")
    print('  --sample-text="Hello World"  # Set sample text for showcase')
    print("  --sample-color=RED           # Set color for samples")
    print("  --sample-fonts=slant,mini    # Specify which fonts to sample")
    print("  --unicode, -u                # Enable Unicode character support")
    print("  --version, -v                # Display version information")
    print("  --help                       # Show help message")
    print()

    # Next steps
    print(divider)
    print("🚀 WHAT'S NEXT?".center(132))
    print(divider)
    print()
    print("1. Try creating your own ASCII art:")
    print("   figlet_forge --font=slant --color=rainbow 'Your Custom Text'")
    print()
    print("2. Explore more options:")
    print("   figlet_forge --help")
    print()
    print("3. Save your creations:")
    print(
        "   figlet_forge --font=big --color=blue_on_black 'Hello' --output=banner.txt"
    )
    print()
    print("4. Combine with other tools:")
    print("   figlet_forge 'Welcome' | lolcat")
    print("   echo 'Hello' | figlet_forge --font=mini --border=rounded")
    print()
    print("5. Create a login banner:")
    print(
        "   figlet_forge --font=big --color=green_on_black --border=double 'SYSTEM LOGIN' > /etc/motd"
    )
    print()
    print(f"⚛️ {PROGRAM_NAME} v{VERSION} ⚡ - Eidosian Typography Engine")
    print('  "Form follows function; elegance emerges from precision."')


# Aliases for backward compatibility
class ColorShowcase:
    """Color showcase class for compatibility."""
    
    def __init__(self, font: str = "small", width: int = 80):
        self.font = font
        self.width = width
    
    @eidosian()
    def display(self, sample_text: str = "hello") -> None:
        """Display color showcase."""
        show_color_showcase(sample_text, self.font, self.width)
    
    @eidosian()
    def generate(self, sample_text: str = "hello") -> str:
        """Generate showcase text (returns empty for now)."""
        return ""


@eidosian()
def display_color_showcase(
    sample_text: str = "hello",
    font: str = "small",
    width: int = 80,
) -> None:
    """Alias for show_color_showcase."""
    show_color_showcase(sample_text, font, width)


@eidosian()
def generate_showcase(
    sample_text: str = "hello",
    font: str = "small",
    width: int = 80,
) -> str:
    """Generate showcase output as string."""
    import io
    import sys
    
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    try:
        show_font_showcase(sample_text, font, width)
    finally:
        sys.stdout = old_stdout
    
    return buffer.getvalue()
