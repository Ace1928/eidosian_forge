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
    title = "⚛️ FIGLET FORGE COLOR SHOWCASE ⚡"

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

    # Get color names as a list
    color_names = list(COLORS.keys())
    basic_colors = color_names[:8]
    
    # Showcase basic colors
    print("Basic colors:")
    print("-------------")
    print()
    _show_colors(rendered, basic_colors)

    # Showcase bright colors
    print("Bright colors:")
    print("--------------")
    print()
    _show_colors(rendered, [f"LIGHT_{c}" for c in basic_colors if not c.startswith("LIGHT_")])

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


# Color styling functions for showcase
def rainbow_colorize(text: str) -> str:
    """Apply rainbow coloring to text."""
    colors = [
        "\033[91m",  # Red
        "\033[93m",  # Yellow
        "\033[92m",  # Green
        "\033[96m",  # Cyan
        "\033[94m",  # Blue
        "\033[95m",  # Magenta
    ]
    result = []
    color_idx = 0
    for char in text:
        if char != ' ' and char != '\n':
            result.append(colors[color_idx % len(colors)] + char)
            color_idx += 1
        else:
            result.append(char)
    return "".join(result) + "\033[0m"


def gradient_colorize(text: str, start_color: str = "red", end_color: str = "blue") -> str:
    """Apply gradient coloring to text."""
    # Simple gradient simulation
    return f"\033[91m{text}\033[0m"  # Fallback to red


def color_style_apply(text: str, style: str) -> str:
    """Apply a named color style to text."""
    styles = {
        "RED": "\033[91m",
        "GREEN": "\033[92m",
        "BLUE": "\033[94m",
        "YELLOW": "\033[93m",
        "CYAN": "\033[96m",
        "MAGENTA": "\033[95m",
    }
    color = styles.get(style.upper(), "")
    return f"{color}{text}\033[0m" if color else text


class ColorShowcase:
    """
    Full-featured color showcase class.
    
    Provides methods for generating font and color showcases with
    customizable options and terminal color support detection.
    """
    
    # Class-level color categories
    _color_categories = {
        "Basic": ["RED", "GREEN", "BLUE", "YELLOW", "CYAN", "MAGENTA", "WHITE", "BLACK"],
        "Bright": ["LIGHT_RED", "LIGHT_GREEN", "LIGHT_BLUE", "LIGHT_YELLOW", "LIGHT_CYAN", "LIGHT_MAGENTA"],
        "Effects": ["RAINBOW", "RED_TO_BLUE", "YELLOW_TO_GREEN", "MAGENTA_TO_CYAN"],
        "Backgrounds": ["WHITE_ON_BLACK", "BLACK_ON_WHITE", "GREEN_ON_BLACK", "BLUE_ON_BLACK"],
    }
    
    # Default fonts for showcase
    _default_fonts = ["standard", "slant", "small", "mini", "big"]
    
    # Font descriptions
    _font_descriptions = {
        "standard": "The classic figlet font, perfect for general use",
        "slant": "Adds a dynamic, forward-leaning style to text",
        "small": "Compact representation when space is limited",
        "mini": "Ultra-compact font for constrained spaces",
        "big": "Bold, attention-grabbing display for headlines",
        "banner": "Wide, horizontally stretched text for announcements",
        "block": "Solid, impactful lettering for emphasis",
        "bubble": "Rounded, friendly letters for approachable messages",
        "digital": "Technical, LCD-like display for a technological feel",
        "shadow": "Dimensional text with drop shadows for depth",
        "script": "Elegant, cursive-like appearance for sophistication",
        "smslant": "Small slanted font for compact, dynamic text",
        "smscript": "Small script font for elegant, compact text",
        "lean": "Condensed characters for fitting more text horizontally",
        "ivrit": "Right-to-left oriented font for Hebrew-style text",
    }
    
    def __init__(self, use_color: bool = True, font: str = "small", width: int = 80):
        """
        Initialize ColorShowcase.
        
        Args:
            use_color: Whether to enable color output
            font: Default font for showcases
            width: Output width
        """
        self.use_color = use_color
        self.font = font
        self.width = width
        self.color_styles = self._color_categories.copy()
        self.font_descriptions = self._font_descriptions.copy()
        # Determine terminal color capability at init time
        import sys
        try:
            self._terminal_supports_color = sys.stdout.isatty() if hasattr(sys.stdout, 'isatty') else False
        except (OSError, AttributeError):
            self._terminal_supports_color = False
    
    def _color_code(self, code: str) -> str:
        """Return color code if colors enabled and terminal supports them."""
        if self.use_color and self._terminal_supports_color:
            return code
        return ""
    
    @eidosian()
    def print_header(self, text: str) -> None:
        """Print a styled header."""
        cyan = self._color_code("\033[1;36m")
        reset = self._color_code("\033[0m")
        print(f"{cyan}{'=' * 80}{reset}")
        print(f"{cyan}{text.center(80)}{reset}")
        print(f"{cyan}{'=' * 80}{reset}")
    
    @eidosian()
    def print_subheader(self, text: str) -> None:
        """Print a styled subheader."""
        yellow = self._color_code("\033[1;33m")
        reset = self._color_code("\033[0m")
        print(f"{yellow}{'─' * 50}{reset}")
        print(f"{yellow}  FONT: {text}{reset}")
        print(f"{yellow}{'─' * 50}{reset}")
    
    @eidosian()
    def print_success(self, text: str) -> None:
        """Print a success message."""
        green = self._color_code("\033[92m")
        reset = self._color_code("\033[0m")
        print(f"{green}✓ {text}{reset}")
    
    @eidosian()
    def print_info(self, text: str) -> None:
        """Print an info message."""
        blue = self._color_code("\033[94m")
        reset = self._color_code("\033[0m")
        print(f"{blue}ℹ {text}{reset}")
    
    @eidosian()
    def print_warning(self, text: str) -> None:
        """Print a warning message."""
        yellow = self._color_code("\033[93m")
        reset = self._color_code("\033[0m")
        print(f"{yellow}⚠ {text}{reset}")
    
    @eidosian()
    def print_error(self, text: str) -> None:
        """Print an error message."""
        red = self._color_code("\033[91m")
        reset = self._color_code("\033[0m")
        print(f"{red}✗ {text}{reset}")
    
    @eidosian()
    def generate_font_showcase(
        self,
        fonts: Optional[List[str]] = None,
        sample_text: str = "hello",
        sample_color: Optional[str] = None,
    ) -> None:
        """
        Generate a font showcase displaying text in various fonts.
        
        Args:
            fonts: List of fonts to showcase, or "ALL" for all fonts, or None for defaults
            sample_text: Text to render in each font
            sample_color: Color to apply (None, color name, "ALL", or "rainbow")
        """
        # Handle fonts parameter
        if fonts == "ALL":
            try:
                fonts = Figlet.getFonts()[:15]  # Limit to 15 fonts
            except Exception:
                fonts = self._default_fonts
        elif fonts is None or fonts == [] or fonts == "":
            fonts = self._default_fonts
        
        # Print header
        self.print_header(f"FIGLET FORGE FONT SHOWCASE - {sample_text.upper()}")
        print()
        
        for font_name in fonts:
            self.print_subheader(font_name)
            print()
            
            try:
                fig = Figlet(font=font_name, width=self.width)
                rendered = fig.render_text(sample_text)
                
                # Apply color if specified
                if sample_color and sample_color != "ALL":
                    if sample_color.lower() == "rainbow":
                        try:
                            rendered = rainbow_colorize(str(rendered))
                        except Exception as e:
                            self.print_warning(f"Warning: Could not apply color rainbow: {e}")
                    else:
                        try:
                            fg, bg = parse_color(sample_color)
                            rendered = f"{fg}{bg}{rendered}\033[0m"
                        except Exception as e:
                            self.print_warning(f"Warning: Could not apply color {sample_color}: {e}")
                
                print(rendered)
                
                # Show description if available
                desc = self.font_descriptions.get(font_name, "")
                if desc:
                    self.print_info(desc)
                
            except Exception as e:
                self.print_error(f"Error loading font '{font_name}': {e}")
            
            print()
    
    @eidosian()
    def generate_usage_guide(self) -> None:
        """Generate the usage guide section."""
        self.print_header("FIGLET FORGE USAGE GUIDE")
        print()
        
        print("📋 CORE USAGE PATTERNS:")
        print("  figlet_forge 'Your Text Here'              # Simple rendering")
        print("  cat file.txt | figlet_forge                # Pipe text from stdin")
        print()
        
        print("🔤 FONT METAMORPHOSIS:")
        print("  figlet_forge --font=slant 'Text'           # Use slant font")
        print("  figlet_forge --font=big 'Text'             # Use big font")
        print("  figlet_forge --list-fonts                  # List all fonts")
        print()
        
        print("🎨 COLOR EFFECTS:")
        print("  figlet_forge --color=RED 'Text'            # Basic color")
        print("  figlet_forge --color=rainbow 'Text'        # Rainbow effect")
        print("  figlet_forge --color=red_to_blue 'Text'    # Gradient effect")
        print()
        
        print("📐 SPATIAL ARCHITECTURE:")
        print("  --width=120                  # Set output width")
        print("  --justify=center             # Center-align text")
        print("  --reverse                    # Mirror text")
        print("  --flip                       # Flip text")
        print()
    
    @classmethod
    def get_color_categories(cls) -> dict:
        """
        Get available color categories.
        
        Returns:
            Dictionary mapping category names to lists of colors
        """
        return cls._color_categories.copy()
    
    @classmethod
    @eidosian()
    def generate_color_showcase(cls, sample_text: str = "hello") -> str:
        """
        Generate color showcase as a string (class method).
        
        Args:
            sample_text: Text to render
            
        Returns:
            String containing the color showcase output
        """
        import io
        import sys
        
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        try:
            showcase = cls(use_color=False)
            showcase.print_header("FIGLET FORGE COLOR SHOWCASE")
            print()
            print(f"Sample: {sample_text}")
            
            # Show basic color categories
            for category, colors in cls._color_categories.items():
                print(f"\n{category}:")
                for color in colors[:4]:
                    print(f"  - {color}")
        finally:
            sys.stdout = old_stdout
        
        return buffer.getvalue()
    
    @eidosian()
    def display_color_showcase_instance(self, font: str = "small", sample_text: str = "hello") -> None:
        """
        Display a color showcase (instance method).
        
        Args:
            font: Font to use for the showcase
            sample_text: Text to render
        """
        self.print_header("FIGLET FORGE COLOR SHOWCASE")
        print()
        
        try:
            fig = Figlet(font=font, width=self.width)
            rendered = str(fig.render_text(sample_text))
        except Exception as e:
            self.print_error(f"Error loading font: {e}")
            return
        
        # Show rainbow effect
        print("RAINBOW EFFECT:")
        print(rainbow_colorize(rendered))
        print()
        
        # Show gradient effect
        print("GRADIENT EFFECT:")
        print(gradient_colorize(rendered))
        print()
        
        # Show basic colors
        for category, colors in self._color_categories.items():
            print(f"{category.upper()} COLORS:")
            for color in colors[:3]:  # Show first 3 of each category
                try:
                    colored = color_style_apply(rendered, color)
                    print(f"  {color}:")
                    print(colored)
                except Exception:
                    print(f"  {color}: (unavailable)")
            print()
    
    # Alias for backward compatibility
    display = display_color_showcase_instance


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
    fonts: Optional[List[str]] = None,
    color: Optional[str] = None,
    width: int = 80,
) -> str:
    """
    Generate showcase output as string.
    
    Args:
        sample_text: Text to use in the showcase
        fonts: List of fonts to showcase, or "ALL" for all, or None for defaults
        color: Color to apply (None, color name, "ALL", or "rainbow")
        width: Output width
        
    Returns:
        String containing the showcase output
    """
    import io
    import sys
    
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    try:
        showcase = ColorShowcase(use_color=True, width=width)
        # Call with positional args as tests expect
        showcase.generate_font_showcase(fonts, sample_text, color)
        showcase.print_header("END OF SHOWCASE")
        showcase.generate_usage_guide()
    finally:
        sys.stdout = old_stdout
    
    return buffer.getvalue()
