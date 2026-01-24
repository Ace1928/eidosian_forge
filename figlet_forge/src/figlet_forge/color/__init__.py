from eidosian_core import eidosian
"""
Color utilities for Figlet text.

This module provides functions and classes to add color to ASCII art text.
"""

# Define the color formats dict for the CLI
color_formats = {
    "rainbow": "Apply rainbow colors to text",
    "gradient": "Apply gradient colors (requires start/end colors)",
    "random": "Apply random colors to text",
    "pulse": "Apply pulsing effect with a single color",
    "red_on_black": "Red text on black background",
    "green_on_black": "Green text on black background",
    "yellow_on_blue": "Yellow text on blue background",
    "white_on_red": "White text on red background",
    "cyan_on_black": "Cyan text on black background",
}

from typing import Callable, Optional, Union

from .effects import (
    color_style_apply,
    gradient_colorize,
    highlight_pattern,
    pulse_colorize,
    rainbow_colorize,
    random_colorize,
)
from .figlet_color import (
    COLOR_CODES,
    RESET_COLORS,
    ColorMode,
    ColorScheme,
    color_to_ansi,
    colored_format,
    get_coloring_functions,
    parse_color,
)


@eidosian()
def colored_format(
    text: str, font: Optional[str] = None, color: Optional[str] = None, **kwargs
) -> str:
    """
    Format text as ASCII art with color.

    Args:
        text: Text to format
        font: Font to use
        color: Color specification
        **kwargs: Additional arguments for Figlet

    Returns:
        Colored ASCII art
    """
    # Import here to avoid circular imports
    from ..figlet import Figlet

    # Create Figlet instance and render text
    fig = Figlet(font=font, **kwargs)
    rendered = fig.renderText(text)

    # Apply color if specified
    if color:
        coloring_func = get_coloring_functions(color)
        if coloring_func:
            rendered = coloring_func(rendered)

    return rendered


@eidosian()
def get_coloring_functions(color_spec: str) -> Optional[Callable[[str], str]]:
    """
    Get the appropriate coloring function based on the specification.

    Args:
        color_spec: Color specification (name, gradient, effect)

    Returns:
        Function that applies the color to a string, or None if invalid
    """
    # Predefined styles
    if color_spec.lower() in [
        "rainbow",
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
        "metal",
        "fire",
        "ice",
        "neon",
        "rgb",
    ]:
        return lambda text: color_style_apply(text, color_spec.lower())

    # Rainbow special case
    if color_spec.upper() == "RAINBOW":
        return rainbow_colorize

    # Gradient
    if "_to_" in color_spec.lower():
        colors = color_spec.lower().split("_to_")
        if len(colors) == 2:
            return lambda text: gradient_colorize(text, colors[0], colors[1])

    # Pulse effect
    if color_spec.lower().startswith("pulse_"):
        base_color = color_spec[6:]
        return lambda text: pulse_colorize(text, base_color)

    # Random colors
    if color_spec.lower() == "random":
        return random_colorize

    # Basic color (handled by parse_color)
    try:
        fg_code, bg_code = parse_color(color_spec)
        return lambda text: apply_color(text, fg_code, bg_code)
    except Exception:
        # Return None for invalid color specs
        return None


@eidosian()
def apply_color(text: str, fg_code: str, bg_code: str = "") -> str:
    """
    Apply ANSI color codes to text.

    Args:
        text: Text to color
        fg_code: Foreground color code
        bg_code: Background color code

    Returns:
        Colored text
    """
    from .figlet_color import RESET_COLORS

    colored_lines = []
    for line in text.splitlines():
        if line.strip():  # Only color non-empty lines
            colored_lines.append(f"{fg_code}{bg_code}{line}{RESET_COLORS}")
        else:
            colored_lines.append(line)

    return "\n".join(colored_lines)


__all__ = [
    "color_to_ansi",
    "parse_color",
    "colored_format",
    "ColorScheme",
    "ColorMode",
    "rainbow_colorize",
    "gradient_colorize",
    "random_colorize",
    "pulse_colorize",
    "highlight_pattern",
    "color_style_apply",
    "color_formats",
    "get_coloring_functions",
    "COLOR_CODES",
    "RESET_COLORS",
]
