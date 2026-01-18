"""
Sample generation functionality for Figlet Forge.

This module provides utilities for generating and displaying samples of
fonts and color combinations, following Eidosian principles of clarity,
elegance, and comprehensive exploration.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..color.effects import gradient_colorize, rainbow_colorize
from ..color.figlet_color import COLOR_CODES
from ..core.figlet_font import FigletFont
from ..figlet import Figlet
from ..version import __version__

# Default sample text with meaningful structure
DEFAULT_SAMPLE_TEXT = "Welcome\nTo The\nEidosian\nFiglet\nForge!"

# Limit for number of fonts to sample in one go (prevent overwhelming output)
MAX_FONTS_SAMPLE = 100

# Cache directory for samples
CACHE_DIR = Path.home() / ".figlet_forge" / "cache"

# Font usage hints - standardized descriptions of when to use specific fonts
FONT_USAGE_HINTS = {
    # Display fonts
    "standard": "The classic figlet font, perfect for general use",
    "slant": "Adds a dynamic, forward-leaning style to text",
    "small": "Compact representation when space is limited",
    "big": "Bold, attention-grabbing display for headlines",
    "mini": "Ultra-compact font for constrained spaces",
    "block": "Solid, impactful lettering for emphasis",
    "banner": "Wide, horizontally stretched text for announcements",
    "shadow": "Dimensional text with drop shadows for depth",
    # Script-like fonts
    "script": "Elegant, cursive-like appearance for a sophisticated look",
    "smscript": "Small script font for elegant, compact text",
    "smslant": "Small slanted font for compact, dynamic text",
    # Technical fonts
    "digital": "Technical, LCD-like display for a technological feel",
    "binary": "Text styled like binary code for tech themes",
    "hex": "Hexadecimal-inspired font for programming contexts",
    # Stylized fonts
    "bubble": "Rounded, friendly letters for approachable messaging",
    "lean": "Condensed characters for fitting more text horizontally",
    "ivrit": "Right-to-left oriented font for Hebrew-style text",
    "smshadow": "Small shadow font for subtle dimensional effects",
    "term": "Terminal-friendly font that works well in console outputs",
    # Add more fonts as needed
}

# Default hint for fonts without specific usage hints
DEFAULT_FONT_HINT = "A specialized decorative font for creative typography"

# Color combination descriptions with standardized snake_case naming
COLOR_COMBINATIONS = [
    ("RED", "BLUE", "red_to_blue", "Vibrant transition from warm to cool"),
    ("YELLOW", "GREEN", "yellow_to_green", "Nature-inspired sunny to leafy gradient"),
    ("MAGENTA", "CYAN", "magenta_to_cyan", "Striking contrast with high visibility"),
    ("WHITE", "BLUE", "white_to_blue", "Clean fade to cool blue depths"),
]

# Foreground/background pairs with standardized snake_case naming
FG_BG_PAIRS = [
    ("RED", "BLACK", "red_on_black", "Classic high-contrast combination"),
    ("GREEN", "BLACK", "green_on_black", "Terminal/matrix style"),
    ("YELLOW", "BLUE", "yellow_on_blue", "High visibility warning style"),
    ("WHITE", "RED", "white_on_red", "Bold alert messaging"),
    ("BLACK", "WHITE", "black_on_white", "Traditional print style"),
    ("CYAN", "BLACK", "cyan_on_black", "Retro computer terminal look"),
]


def ensure_cache_dir() -> Path:
    """Ensure the cache directory exists and return its path."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def get_cache_path(sample_type: str, text: str) -> Path:
    """
    Get the cache file path for a specific sample type and text.

    Args:
        sample_type: Type of sample ('font' or 'color')
        text: The text used in the sample

    Returns:
        Path to the cache file
    """
    # Create a deterministic filename based on content
    text_hash = str(hash(text))[:10]  # Use part of hash for filename
    return ensure_cache_dir() / f"sample_{sample_type}_{text_hash}.json"


def load_from_cache(sample_type: str, text: str) -> Optional[Dict]:
    """
    Try to load sample data from cache.

    Args:
        sample_type: Type of sample ('font' or 'color')
        text: The text used in the sample

    Returns:
        Cached sample data or None if not available/valid
    """
    cache_path = get_cache_path(sample_type, text)

    if not cache_path.exists():
        return None

    try:
        # Check if cache is recent (less than 1 day old)
        if time.time() - cache_path.stat().st_mtime > 86400:
            return None

        with open(cache_path) as f:
            data = json.load(f)

        # Validate cache structure
        if not isinstance(data, dict) or "samples" not in data:
            return None

        return data
    except (json.JSONDecodeError, OSError):
        # Invalid cache, ignore it
        return None


def save_to_cache(sample_type: str, text: str, data: Dict) -> bool:
    """
    Save sample data to cache.

    Args:
        sample_type: Type of sample ('font' or 'color')
        text: The text used in the sample
        data: The sample data to cache

    Returns:
        True if saved successfully, False otherwise
    """
    cache_path = get_cache_path(sample_type, text)

    try:
        with open(cache_path, "w") as f:
            json.dump(data, f)
        return True
    except (OSError, TypeError):
        return False


def generate_font_samples(
    text: str = DEFAULT_SAMPLE_TEXT,
    max_fonts: int = MAX_FONTS_SAMPLE,
    fonts: Optional[List[str]] = None,
    width: int = 80,
) -> Tuple[Dict[str, str], List[str]]:
    """
    Generate text samples in all available fonts.

    Args:
        text: Text to render in different fonts
        max_fonts: Maximum number of fonts to sample
        fonts: Optional list of specific fonts to sample
        width: Width for rendering

    Returns:
        Tuple of (dictionary mapping font names to rendered samples, list of failed fonts)
    """
    # Try to load from cache first
    cache = load_from_cache("font", text)
    if cache:
        return cache["samples"], cache.get("failed_fonts", [])

    # Get available fonts
    available_fonts = fonts or FigletFont.getFonts()
    total_fonts = len(available_fonts)

    # Print summary header
    print(f"\n\033[1;36m{'═' * 80}\033[0m")
    print(
        f"\033[1;36m FONT SHOWCASE: Preparing to render text in {min(total_fonts, max_fonts)} fonts\033[0m"
    )
    print(f"\033[1;36m{'═' * 80}\033[0m\n")

    # Limit number of fonts to prevent overwhelming output
    if len(available_fonts) > max_fonts:
        print(
            f"\033[1;33mLimiting to {max_fonts} fonts out of {total_fonts} available\033[0m"
        )
        available_fonts = available_fonts[:max_fonts]

    samples = {}
    failed_fonts = []
    processed = 0

    # Generate samples for each font
    for font_name in available_fonts:
        processed += 1
        progress = processed * 100 // len(available_fonts)
        print(
            f"\r\033[KProcessing fonts: {progress}% complete ({processed}/{len(available_fonts)}) - Current: {font_name}",
            end="",
        )

        try:
            fig = Figlet(font=font_name, width=width)
            rendered = fig.renderText(text)
            samples[font_name] = rendered
            # Print clear success indicator
            print(f"\r\033[K\033[32m✓\033[0m Font loaded successfully: {font_name}")
        except Exception as e:
            # Record fonts that failed to load
            failed_fonts.append(font_name)
            error_msg = str(e)
            # Truncate very long error messages
            if len(error_msg) > 80:
                error_msg = error_msg[:77] + "..."
            print(
                f"\r\033[K\033[31m✗\033[0m Failed to load font: {font_name} - {error_msg}"
            )
            continue

    print(
        f"\r\033[K\033[1;32mCompleted font processing: {len(samples)} succeeded, {len(failed_fonts)} failed\033[0m"
    )

    # Cache the results
    save_to_cache("font", text, {"samples": samples, "failed_fonts": failed_fonts})

    return samples, failed_fonts


def generate_color_matrix(
    text: str = "Figlet",
    font: str = "standard",
    width: int = 80,
) -> Dict[str, str]:
    """
    Generate a matrix of color combinations for the given text.

    Args:
        text: Text to render with colors
        font: Font to use for rendering
        width: Width for rendering

    Returns:
        Dictionary mapping color descriptions to colored samples
    """
    # Try to load from cache first
    cache_key = f"{text}_{font}"
    cache = load_from_cache("color", cache_key)
    if cache:
        return cache["samples"]

    # Generate colored samples
    print(f"\n\033[1;36m{'═' * 80}\033[0m")
    print(
        f"\033[1;36m COLOR SHOWCASE: Generating color combinations for '{text}' using '{font}' font\033[0m"
    )
    print(f"\033[1;36m{'═' * 80}\033[0m\n")

    try:
        # Try to load the specified font
        print(f"\033[KAttempting to load font: {font}...")
        fig = Figlet(font=font, width=width)
        print(f"\033[K\033[32m✓\033[0m Font '{font}' loaded successfully")
        plain_text = fig.renderText(text)

        samples = {}

        # Add rainbow sample
        print("\033[KGenerating rainbow color effect...")
        samples["rainbow"] = rainbow_colorize(plain_text)

        # Add gradient samples with standardized names
        for start_color, end_color, name, desc in COLOR_COMBINATIONS:
            print(f"\033[KGenerating gradient: {name} ({desc})...")
            samples[name] = gradient_colorize(plain_text, start_color, end_color)

        # Add foreground/background combinations
        for fg, bg, name, desc in FG_BG_PAIRS:
            print(f"\033[KGenerating color style: {name} ({desc})...")
            # Create ANSI color codes
            fg_code = f"\033[{COLOR_CODES[fg]}m" if fg in COLOR_CODES else ""
            bg_code = f"\033[{int(COLOR_CODES[bg]) + 10}m" if bg in COLOR_CODES else ""
            reset_code = "\033[0m"

            # Apply colors
            samples[name] = f"{fg_code}{bg_code}{plain_text}{reset_code}"

        print(
            f"\033[K\033[1;32mCompleted color matrix with {len(samples)} styles\033[0m"
        )

        # Cache the results
        save_to_cache("color", cache_key, {"samples": samples})

        return samples

    except Exception as e:
        print(f"\033[K\033[1;31mError generating color matrix: {str(e)}\033[0m")
        print(
            "\033[K\033[1;33mFalling back to 'standard' font for color samples\033[0m"
        )

        try:
            # Fallback to standard font
            fallback_font = "standard"
            fig = Figlet(font=fallback_font, width=width)
            plain_text = fig.renderText(text)

            # Create minimal samples with fallback
            samples = {
                "rainbow": rainbow_colorize(plain_text),
                "red_to_blue": gradient_colorize(plain_text, "RED", "BLUE"),
                "green_on_black": f"\033[{COLOR_CODES['GREEN']}m\033[{int(COLOR_CODES['BLACK']) + 10}m{plain_text}\033[0m",
            }

            print(
                "\033[K\033[1;32mCreated limited color samples with fallback font\033[0m"
            )
            return samples

        except Exception:
            print(
                "\033[K\033[1;31mCritical error: Could not generate any color samples\033[0m"
            )
            # Return an empty dict if we can't generate colors
            return {}


def display_font_sample(
    font_name: str, sample: str, with_color: bool = False, interactive: bool = False
) -> None:
    """
    Display a font sample with standardized formatting.

    Args:
        font_name: Name of the font
        sample: Rendered text sample
        with_color: Whether to add color to the header
        interactive: Whether to wait for user input between samples
    """
    # Create a visually distinctive header
    header_width = min(
        80, max(len(font_name) + 10, len(sample.splitlines()[0]) if sample else 20)
    )

    # Add color to header if requested
    header_start = "\033[1;36m" if with_color else ""  # Bright cyan, bold
    header_end = "\033[0m" if with_color else ""

    # Get usage hint if available
    usage_hint = FONT_USAGE_HINTS.get(font_name.lower(), DEFAULT_FONT_HINT)

    # Print the header with standardized format
    print(f"\n{header_start}{'═' * header_width}")
    print(f" FONT: {font_name}")
    print(f" INFO: {usage_hint}")
    print(f" USAGE: figlet_forge --font={font_name} 'Your Text'")
    print(f"{'═' * header_width}{header_end}\n")

    # Print the sample
    print(sample)

    # In interactive mode, wait for user to press Enter
    if interactive:
        input("\n\033[1;33mPress Enter for next font...\033[0m")


def display_color_sample(
    color_name: str, sample: str, description: str = "", interactive: bool = False
) -> None:
    """
    Display a color sample with standardized formatting.

    Args:
        color_name: Description of the color
        sample: Colored text sample
        description: Additional description of the color style
        interactive: Whether to wait for user input between samples
    """
    # Print header and sample with standardized format
    header_width = min(80, max(len(color_name) + 20, len(description) + 10))

    print(f"\n\033[1;35m{'═' * header_width}")  # Magenta header
    print(f" COLOR STYLE: {color_name}")
    if description:
        print(f" INFO: {description}")
    print(f" USAGE: figlet_forge --color={color_name} 'Your Text'")
    print(f"{'═' * header_width}\033[0m\n")

    # Print the colored sample
    print(sample)

    # In interactive mode, wait for user to press Enter
    if interactive:
        input("\n\033[1;33mPress Enter for next color style...\033[0m")


def display_usage_guide(
    font_samples: Dict[str, str], color_samples: Dict[str, str]
) -> None:
    """
    Display a usage guide for the user with clear next steps.

    Args:
        font_samples: Dictionary of font samples that were displayed
        color_samples: Dictionary of color samples that were displayed
    """
    print("\n\033[1;32m" + "═" * 80 + "\033[0m")
    print("\033[1;32m ⚛️ FIGLET FORGE USAGE GUIDE - THE EIDOSIAN WAY ⚡\033[0m")
    print("\033[1;32m" + "═" * 80 + "\033[0m")

    # Summary section
    print("\n\033[1;36m📊 SHOWCASE SUMMARY:\033[0m")
    print(
        f"  • Displayed {len(font_samples)} fonts and {len(color_samples)} color styles"
    )
    print("  • Use the commands below to apply these styles to your own text")

    # Basic usage
    print("\n\033[1m📋 CORE USAGE PATTERNS:\033[0m")
    print("  figlet_forge 'Your Text Here'              # Simple rendering")
    print("  cat file.txt | figlet_forge                # Pipe text from stdin")
    print("  figlet_forge 'Line 1\\nLine 2'              # Multi-line text")

    # Font examples with clear formatting
    if font_samples:
        print("\n\033[1m🔤 FONT METAMORPHOSIS:\033[0m")
        print("  Command format: figlet_forge --font=<font_name> 'Your Text'")
        print("\n  Available fonts you've just seen:")

        # Group fonts by category for better organization
        categories = {
            "Display Fonts": ["big", "block", "banner", "shadow", "standard"],
            "Script Fonts": ["script", "slant", "smscript"],
            "Compact Fonts": ["small", "mini", "smslant"],
            "Technical Fonts": ["digital", "binary", "hex"],
            "Stylized Fonts": ["bubble", "ivrit", "lean"],
        }

        # Organize fonts by category
        categorized_fonts = {}
        uncategorized = []

        for font in font_samples.keys():
            found = False
            for category, font_list in categories.items():
                if font.lower() in font_list:
                    if category not in categorized_fonts:
                        categorized_fonts[category] = []
                    categorized_fonts[category].append(font)
                    found = True
                    break

            if not found:
                uncategorized.append(font)

        # Display fonts by category
        for category, fonts in categorized_fonts.items():
            if fonts:
                print(f"\n    {category}:")
                for font in fonts:
                    hint = FONT_USAGE_HINTS.get(font.lower(), DEFAULT_FONT_HINT)
                    # Truncate hint if it's too long
                    if len(hint) > 50:
                        hint = hint[:47] + "..."
                    print(f"      figlet_forge --font={font} 'Your Text'  # {hint}")

        # Display uncategorized fonts (limited to 10 for brevity)
        if uncategorized:
            print("\n    Other Fonts:")
            for font in uncategorized[:10]:
                print(f"      figlet_forge --font={font} 'Your Text'")

            if len(uncategorized) > 10:
                print(f"      ... and {len(uncategorized) - 10} more fonts")

    # Color examples with clear formatting
    if color_samples:
        print("\n\033[1m🎨 CHROMATIC TRANSFORMATIONS:\033[0m")
        print("  Command format: figlet_forge --color=<color_style> 'Your Text'")
        print("\n  Available color styles you've just seen:")

        # Rainbow style first (if available)
        if "rainbow" in color_samples:
            print(
                "    • figlet_forge --color=rainbow 'Your Text'  # Dynamic multi-color effect"
            )

        # Show gradient styles
        gradients_shown = False
        for _, _, name, desc in COLOR_COMBINATIONS:
            if name in color_samples:
                if not gradients_shown:
                    print("\n    Gradient styles:")
                    gradients_shown = True
                print(f"    • figlet_forge --color={name} 'Your Text'  # {desc}")

        # Show foreground/background styles
        fg_bg_shown = False
        for _, _, name, desc in FG_BG_PAIRS:
            if name in color_samples:
                if not fg_bg_shown:
                    print("\n    Foreground/background styles:")
                    fg_bg_shown = True
                print(f"    • figlet_forge --color={name} 'Your Text'  # {desc}")

        print("\n    Custom RGB colors:")
        print(
            "    • figlet_forge --color=255;0;0 'Your Text'      # RGB red foreground"
        )
        print(
            "    • figlet_forge --color=:0;0;255 'Your Text'     # RGB blue background"
        )
        print("    • figlet_forge --color=255;0;0:0;0;255 'Your Text'  # RGB fg:bg")

    # Layout & alignment options
    print("\n\033[1m📐 SPATIAL ARCHITECTURE:\033[0m")
    print("  --width=120                  # Set output width")
    print("  --justify=center             # Center-align text (left, right, center)")
    print("  --direction=right-to-left    # Change text direction")

    # Text transformations
    print("\n\033[1m🔄 RECURSIVE TRANSFORMATIONS:\033[0m")
    print("  --reverse                    # Mirror text horizontally")
    print("  --flip                       # Flip text vertically (upside-down)")
    print(
        "  --border=single              # Add border (single, double, rounded, bold, ascii)"
    )
    print("  --shade                      # Add shadow effect")

    # Combined examples
    print("\n\033[1m✨ SYNTHESIS PATTERNS:\033[0m")
    print("  # Rainbow colored slant font with border")
    print("  figlet_forge --font=slant --color=rainbow --border=single 'Synthesis'")
    print("\n  # Center-justified big green text on black background")
    print("  figlet_forge --font=big --color=green_on_black --justify=center 'Elegant'")
    print("\n  # Flipped and reversed text with shadow effect")
    print("  figlet_forge --font=standard --reverse --flip --shade 'Recursion'")

    # "What's Next" section - clear guidance for users
    print("\n\033[1;33m" + "═" * 80 + "\033[0m")
    print("\033[1;33m 🚀 WHAT'S NEXT?\033[0m")
    print("\033[1;33m" + "═" * 80 + "\033[0m")

    # Provide clear next steps
    print("\n1. Try creating your own ASCII art:")
    print("   figlet_forge --font=slant --color=rainbow 'Your Custom Text'")

    print("\n2. Explore more options:")
    print("   figlet_forge --help")

    print("\n3. Save your creations:")
    print(
        "   figlet_forge --font=big --color=blue_on_black 'Hello' --output=banner.txt"
    )

    print("\n4. Combine with other tools:")
    print("   figlet_forge 'Welcome' | lolcat")  # lolcat is a common colorization tool
    print("   echo 'Hello' | figlet_forge --font=mini --border=rounded")

    print("\n5. Create a login banner:")
    print(
        "   figlet_forge --font=big --color=green_on_black --border=double 'SYSTEM LOGIN' > /etc/motd"
    )

    # Version info with Eidosian flair
    print(
        f"\n\033[1m⚛️ Figlet Forge v{__version__} ⚡\033[0m - Eidosian Typography Engine"
    )
    print('\033[36m  "Form follows function; elegance emerges from precision."\033[0m')


def run_samples(
    text: str = DEFAULT_SAMPLE_TEXT,
    show_fonts: bool = True,
    show_colors: bool = False,
    interactive: bool = False,
    width: int = 80,
    max_fonts: int = MAX_FONTS_SAMPLE,
) -> None:
    """
    Run the sample display process with improved organization and feedback.

    Args:
        text: Text to render in samples
        show_fonts: Whether to show font samples
        show_colors: Whether to show color samples
        interactive: Whether to wait for user input between samples
        width: Width for rendering
        max_fonts: Maximum number of fonts to sample
    """
    # Show welcome message with high-level summary
    print("\n\033[1;32m" + "═" * 80 + "\033[0m")
    print("\033[1;32m FIGLET FORGE SAMPLE SHOWCASE\033[0m")
    print("\033[1;32m" + "═" * 80 + "\033[0m")

    # Show a summary of what will be displayed
    showcase_types = []
    if show_fonts:
        showcase_types.append("fonts")
    if show_colors:
        showcase_types.append("color styles")

    print("\n\033[1;36mSHOWCASE SUMMARY\033[0m")
    print(f"• Exploring {' and '.join(showcase_types)} with Figlet Forge")
    print(f"• Sample text: \"{text.replace('\\n', ' ')}\"")
    print(f"• Width: {width} columns")
    if show_fonts:
        print(f"• Maximum fonts: {max_fonts}")
    print(f"• Interactive mode: {'On' if interactive else 'Off'}")
    print("\nScroll down to see samples and usage instructions...\n")

    font_samples = {}
    color_samples = {}
    displayed_fonts = []
    displayed_colors = []
    failed_fonts = []

    # Generate font samples if requested
    if show_fonts:
        font_samples, failed_fonts = generate_font_samples(
            text, max_fonts=max_fonts, width=width
        )

        # Show summary
        total_fonts = len(font_samples)
        total_failed = len(failed_fonts)

        print("\n\033[1;32m" + "═" * 80 + "\033[0m")
        print("\033[1;32m FONT SHOWCASE SUMMARY\033[0m")
        print("\033[1;32m" + "═" * 80 + "\033[0m")
        print(f"\n✅ Successfully loaded {total_fonts} fonts")
        if failed_fonts:
            print(f"⚠️ {total_failed} fonts failed to load")
            if total_failed > 0 and total_failed <= 5:
                print(f"   Failed fonts: {', '.join(failed_fonts)}")
            elif total_failed > 5:
                print(f"   Failed fonts include: {', '.join(failed_fonts[:5])}...")
                print(f"   ... and {total_failed - 5} more")

        # Show each font sample
        for font_name, sample in font_samples.items():
            displayed_fonts.append(font_name)

            # If showing both fonts and colors, generate color samples for each font
            if show_colors:
                # Show the plain font first
                display_font_sample(
                    font_name, sample, with_color=True, interactive=False
                )

                # Then show color variants
                try:
                    color_matrix = generate_color_matrix(
                        text=(
                            text.splitlines()[0] if "\n" in text else text
                        ),  # Use first line for color samples
                        font=font_name,
                        width=width,
                    )

                    # First show rainbow effect if available
                    if "rainbow" in color_matrix:
                        displayed_colors.append("rainbow")
                        display_color_sample(
                            f"{font_name} - rainbow",
                            color_matrix["rainbow"],
                            description="Dynamic multi-color effect",
                            interactive=False,
                        )

                    # Then show gradient effects
                    for _, _, name, description in COLOR_COMBINATIONS:
                        if name in color_matrix:
                            displayed_colors.append(name)
                            display_color_sample(
                                f"{font_name} - {name}",
                                color_matrix[name],
                                description=description,
                                interactive=False,
                            )

                    # Finally show foreground/background combinations
                    for _, _, name, description in FG_BG_PAIRS:
                        if name in color_matrix:
                            displayed_colors.append(name)
                            display_color_sample(
                                f"{font_name} - {name}",
                                color_matrix[name],
                                description=description,
                                interactive=False,
                            )

                    # Wait after all color variants are shown for this font
                    if interactive:
                        input(
                            "\n\033[1;33mPress Enter for next font with colors...\033[0m"
                        )
                except Exception as e:
                    print(
                        f"\n\033[1;31mError generating color samples for {font_name}: {e}\033[0m"
                    )
                    print(
                        "\033[1;33mSkipping color samples for this font and continuing...\033[0m"
                    )
            else:
                # Just show the font without colors
                display_font_sample(
                    font_name, sample, with_color=True, interactive=interactive
                )

    elif show_colors:
        # If only showing colors without font samples, use standard font
        try:
            print("\n\033[1;35m" + "═" * 80 + "\033[0m")
            print(
                "\033[1;35m COLOR SHOWCASE: Exploring color styles with 'standard' font\033[0m"
            )
            print("\033[1;35m" + "═" * 80 + "\033[0m\n")

            color_samples = generate_color_matrix(
                text=text, font="standard", width=width
            )

            # Rainbow effect
            if "rainbow" in color_samples:
                displayed_colors.append("rainbow")
                display_color_sample(
                    "rainbow",
                    color_samples["rainbow"],
                    description="Dynamic multi-color effect",
                    interactive=interactive,
                )

            # Gradients
            print("\n\033[1m GRADIENT COLOR STYLES\033[0m")
            for start_color, end_color, name, description in COLOR_COMBINATIONS:
                if name in color_samples:
                    displayed_colors.append(name)
                    display_color_sample(
                        name,
                        color_samples[name],
                        description=description,
                        interactive=interactive,
                    )

            # FG/BG combinations
            print("\n\033[1m FOREGROUND/BACKGROUND COLOR STYLES\033[0m")
            for fg, bg, name, description in FG_BG_PAIRS:
                if name in color_samples:
                    displayed_colors.append(name)
                    display_color_sample(
                        name,
                        color_samples[name],
                        description=description,
                        interactive=interactive,
                    )
        except Exception as e:
            print(f"\n\033[1;31mError generating color samples: {e}\033[0m")
            print("\033[1;33mAttempting to continue with limited functionality\033[0m")

    # Show closing message
    print("\n\033[1;32m" + "═" * 80 + "\033[0m")
    print("\033[1;32m END OF SHOWCASE\033[0m")
    print("\033[1;32m" + "═" * 80 + "\033[0m")

    # Display usage guide
    display_usage_guide(font_samples, color_samples)


if __name__ == "__main__":
    import sys

    sys.exit(run_samples(interactive=True))
