from eidosian_core import eidosian
"""
Showcase functionality for Figlet Forge.

This module provides functions to demonstrate fonts and styles available
in Figlet Forge through visual examples and samples.
"""

import sys
from typing import List, Optional

from .color.effects import color_style_apply, gradient_colorize, rainbow_colorize
from .figlet import Figlet
from .version import __version__


class ColorShowcase:
    """
    Generate showcases of color styles with different fonts.

    This class creates visual demonstrations of available color effects
    and font combinations to help users choose their preferred style.
    """

    def __init__(self, use_color: bool = True):
        """
        Initialize the showcase generator.

        Args:
            use_color: Whether to use color in the output
        """
        self.use_color = use_color and sys.stdout.isatty()
        self.color_styles = {
            "rainbow": "Dynamic multi-color effect",
            "red_to_blue": "Vibrant transition from warm to cool",
            "yellow_to_green": "Nature-inspired sunny to leafy gradient",
            "magenta_to_cyan": "Striking contrast with high visibility",
            "white_to_blue": "Clean fade to cool blue depths",
            "red_on_black": "Classic high-contrast combination",
            "green_on_black": "Terminal/matrix style",
            "yellow_on_blue": "High visibility warning style",
            "white_on_red": "Bold alert messaging",
            "black_on_white": "Traditional print style",
            "cyan_on_black": "Retro computer terminal look",
        }
        self.font_descriptions = {
            "standard": "The classic figlet font, perfect for general use",
            "slant": "Adds a dynamic, forward-leaning style to text",
            "small": "Compact representation when space is limited",
            "mini": "Ultra-compact font for constrained spaces",
            "big": "Bold, attention-grabbing display for headlines",
            "banner": "Wide, horizontally stretched text for announcements",
            "block": "Solid, impactful lettering for emphasis",
            "bubble": "Rounded, friendly letters for approachable messages",
            "digital": "Technical, LCD-like display for a technological feel",
            "ivrit": "Right-to-left oriented font for Hebrew-style text",
            "lean": "Condensed characters for fitting more text horizontally",
            "script": "Elegant, cursive-like appearance for a sophisticated look",
            "shadow": "Dimensional text with drop shadows for depth",
            "smscript": "Small script font for elegant, compact text",
            "smslant": "Small slanted font for compact, dynamic text",
        }

    @eidosian()
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

    @eidosian()
    def print_subheader(self, text: str) -> None:
        """
        Print a formatted subheader.

        Args:
            text: Subheader text to display
        """
        if self.use_color:
            print(f"\n\033[1;33m{'═' * 50}\033[0m")
            print(f"\033[1;33m {text}\033[0m")
            print(f"\033[1;33m{'═' * 50}\033[0m\n")
        else:
            print(f"\n{'═' * 50}")
            print(f" {text}")
            print(f"{'═' * 50}\n")

    @eidosian()
    def print_success(self, text: str) -> None:
        """
        Print a success message.

        Args:
            text: Success message to display
        """
        if self.use_color:
            print(f"\033[32m✓ {text}\033[0m")
        else:
            print(f"✓ {text}")

    @eidosian()
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

    @eidosian()
    def generate_font_showcase(
        self,
        fonts: Optional[List[str]] = None,
        sample_text: str = "hello",
        sample_color: Optional[str] = None,
    ) -> None:
        """
        Generate a showcase of fonts.

        Args:
            fonts: List of fonts to showcase (if None, uses a curated selection)
            sample_text: Text to render in each font
            sample_color: Color to apply to the sample text
        """
        if not fonts:
            # Use a curated list of fonts
            fonts = ["standard", "slant", "small", "mini", "big"]

        # Print a header for the showcase
        self.print_header(f"⚛️ FIGLET FORGE FONT SHOWCASE - {sample_text.upper()} ⚡")
        self.print_info(f"Sample text: '{sample_text}'")
        if sample_color:
            self.print_info(f"Sample color: {sample_color}")
        self.print_info(f"Showcasing {len(fonts)} fonts\n")

        for font_name in fonts:
            self.print_subheader(f"FONT: {font_name}")

            # Try to load the font
            try:
                print(f"Attempting to load font: {font_name}...")
                fig = Figlet(font=font_name)
                self.print_success(f"Font '{font_name}' loaded successfully")

                # Render the sample text
                result = fig.renderText(sample_text)

                # Apply color if requested
                if sample_color:
                    try:
                        if sample_color.upper() == "RAINBOW":
                            result = rainbow_colorize(str(result))
                        elif "_TO_" in sample_color.upper():
                            colors = sample_color.upper().split("_TO_")
                            if len(colors) == 2:
                                result = gradient_colorize(
                                    str(result), colors[0], colors[1]
                                )
                        else:
                            # Try to apply as a color style
                            try:
                                result = color_style_apply(str(result), sample_color)
                            except ValueError:
                                # Otherwise, use as direct color spec
                                from .color.figlet_color import parse_color
                                from .version import RESET_COLORS

                                fg, bg = parse_color(sample_color)
                                if fg or bg:
                                    result = f"{fg}{bg}{result}{RESET_COLORS}"
                    except Exception as e:
                        print(f"Warning: Could not apply color '{sample_color}': {e}")

                # Print font information
                description = self.font_descriptions.get(
                    font_name, "A specialized decorative font for creative typography"
                )
                print(f" INFO: {description}")
                print(f" USAGE: figlet_forge --font={font_name} 'Your Text'")

                # Print the rendered text
                print()
                print(result)
                print()

                # Generate color showcase for this font if no specific color was requested
                if (
                    not sample_color and font_name == "standard"
                ):  # Only for standard font to keep showcase compact
                    self.generate_color_showcase(font_name, sample_text)

            except Exception as e:
                print(f"Error loading font '{font_name}': {e}")

    @eidosian()
    def generate_color_showcase(self, font_name: str, sample_text: str) -> None:
        """
        Generate a showcase of color effects for a specific font.

        Args:
            font_name: Name of font to use
            sample_text: Text to render
        """
        self.print_header(
            f"COLOR SHOWCASE: Generating color combinations for '{sample_text}' using '{font_name}' font"
        )

        # Load the font
        try:
            print(f"Attempting to load font: {font_name}...")
            fig = Figlet(font=font_name)
            self.print_success(f"Font '{font_name}' loaded successfully")

            # Render the basic text
            base_result = fig.renderText(sample_text)

            # Generate various color effects
            generated_styles = []

            # Rainbow effect
            print("Generating rainbow color effect...")
            try:
                rainbow = rainbow_colorize(base_result)
                generated_styles.append(("rainbow", rainbow))
            except Exception as e:
                print(f"Error generating rainbow effect: {e}")

            # Gradient effects
            for style_name, desc in [
                ("red_to_blue", "Vibrant transition from warm to cool"),
                ("yellow_to_green", "Nature-inspired sunny to leafy gradient"),
                ("magenta_to_cyan", "Striking contrast with high visibility"),
                ("white_to_blue", "Clean fade to cool blue depths"),
            ]:
                print(f"Generating gradient: {style_name} ({desc})...")
                try:
                    start, end = style_name.split("_to_")
                    gradient = gradient_colorize(
                        base_result, start.upper(), end.upper()
                    )
                    generated_styles.append((style_name, gradient))
                except Exception as e:
                    print(f"Error generating gradient {style_name}: {e}")

            # Solid color combinations
            for style_name, desc in [
                ("red_on_black", "Classic high-contrast combination"),
                ("green_on_black", "Terminal/matrix style"),
                ("yellow_on_blue", "High visibility warning style"),
                ("white_on_red", "Bold alert messaging"),
                ("black_on_white", "Traditional print style"),
                ("cyan_on_black", "Retro computer terminal look"),
            ]:
                print(f"Generating color style: {style_name} ({desc})...")
                try:
                    # Apply the color style
                    colored = color_style_apply(base_result, style_name)
                    generated_styles.append((style_name, colored))
                except Exception as e:
                    print(f"Error generating color style {style_name}: {e}")

            print(f"Completed color matrix with {len(generated_styles)} styles")
            print()

            # Display all generated styles
            for style_name, colored_result in generated_styles:
                # Get description
                desc = self.color_styles.get(style_name, "Custom color effect")

                # Display style header
                self.print_subheader(f"COLOR STYLE: {font_name} - {style_name}")
                print(f" INFO: {desc}")
                print(
                    f" USAGE: figlet_forge --font={font_name} --color={style_name} 'Your Text'"
                )
                print()

                # Display the colored result
                print(colored_result)
                print()

        except Exception as e:
            print(f"Error in color showcase for '{font_name}': {e}")

    @eidosian()
    def generate_usage_guide(self) -> None:
        """Generate a comprehensive usage guide with examples."""
        self.print_header("⚛️ FIGLET FORGE USAGE GUIDE - THE EIDOSIAN WAY ⚡")

        print("📊 SHOWCASE SUMMARY:")
        print("  • Displayed fonts and color styles")
        print("  • Use the commands below to apply these styles to your own text")
        print()

        print("📋 CORE USAGE PATTERNS:")
        print("  figlet_forge 'Your Text Here'              # Simple rendering")
        print("  cat file.txt | figlet_forge                # Pipe text from stdin")
        print("  figlet_forge 'Line 1\\nLine 2'              # Multi-line text")
        print()

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

        print("📐 SPATIAL ARCHITECTURE:")
        print("  --width=120                  # Set output width")
        print(
            "  --justify=center             # Center-align text (left, right, center)"
        )
        print("  --direction=right-to-left    # Change text direction")
        print()

        print("🔄 RECURSIVE TRANSFORMATIONS:")
        print("  --reverse                    # Mirror text horizontally")
        print("  --flip                       # Flip text vertically (upside-down)")
        print(
            "  --border=single              # Add border (single, double, rounded, bold, ascii)"
        )
        print("  --shade                      # Add shadow effect")
        print()

        print("🎨 COLOR EFFECTS:")
        print("  --color=RED                  # Basic color")
        print("  --color=RED:BLACK            # Foreground:Background color")
        print("  --color=rainbow              # Rainbow effect")
        print("  --color=red_to_blue          # Gradient effect")
        print("  --color=green_on_black       # Preset color style")
        print("  --color=list                 # Show available colors")
        print()

        print("✨ SYNTHESIS PATTERNS:")
        print("  # Rainbow colored slant font with border")
        print("  figlet_forge --font=slant --color=rainbow --border=single 'Synthesis'")
        print()
        print("  # Center-justified big green text on black background")
        print(
            "  figlet_forge --font=big --color=green_on_black --justify=center 'Elegant'"
        )
        print()
        print("  # Flipped and reversed text with shadow effect")
        print("  figlet_forge --font=standard --reverse --flip --shade 'Recursion'")
        print()

        print("📋 COMMAND LINE OPTIONS:")
        print("  --showcase                   # Show fonts and styles showcase")
        print("  --sample                     # Equivalent to --showcase")
        print('  --sample-text="Hello World"  # Set sample text for showcase')
        print("  --sample-color=RED           # Set color for samples")
        print("  --sample-fonts=slant,mini    # Specify which fonts to sample")
        print("  --unicode, -u                # Enable Unicode character support")
        print("  --version, -v                # Display version information")
        print("  --help                       # Show help message")

        self.print_header("🚀 WHAT'S NEXT?")

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
        print(f"⚛️ Figlet Forge v{__version__} ⚡ - Eidosian Typography Engine")
        print('  "Form follows function; elegance emerges from precision."')


@eidosian()
def generate_showcase(
    sample_text: str = "hello",
    fonts: Optional[List[str]] = None,
    color: Optional[str] = None,
) -> None:
    """
    Generate a showcase of fonts and color styles.

    Args:
        sample_text: Text to render in the showcase
        fonts: List of fonts to showcase (if None, uses a curated selection)
        color: Color to apply to all samples
    """
    showcase = ColorShowcase()
    showcase.generate_font_showcase(fonts, sample_text, color)
    showcase.print_header("END OF SHOWCASE")
    showcase.generate_usage_guide()


if __name__ == "__main__":
    # If run directly, generate a showcase with default settings
    generate_showcase()
