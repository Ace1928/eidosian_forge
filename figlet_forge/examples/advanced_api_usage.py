#!/usr/bin/env python3
"""
Advanced API Usage Examples for Figlet Forge.

This module demonstrates sophisticated usage patterns of the
Figlet Forge library, showcasing its capabilities in a practical context.
"""

import sys
import time
from pathlib import Path
from typing import (
    List,
    Optional,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)

# Add the package to path when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from figlet_forge import Figlet
from figlet_forge.color.effects import (
    gradient_colorize,
    pulse_colorize,
    rainbow_colorize,
)
from figlet_forge.core.figlet_string import FigletString

# Type variable for more precise generic handling
T = TypeVar("T")


# Define structural protocols for precise duck typing
@runtime_checkable
class Borderable(Protocol):
    """Protocol for objects that can be bordered."""

    def border(self, *args: object, **kwargs: object) -> FigletString: ...


@runtime_checkable
class FlipCapable(Protocol):
    """Protocol for objects that can be flipped."""

    def flip(self) -> FigletString: ...


@runtime_checkable
class ReverseCapable(Protocol):
    """Protocol for objects that can be reversed."""

    def reverse(self) -> FigletString: ...


def create_title_banner(title: str, subtitle: Optional[str] = None) -> str:
    """
    Create a professional-looking title banner with optional subtitle.

    Args:
        title: Main title text
        subtitle: Optional subtitle text

    Returns:
        Formatted banner text with decorative elements
    """
    # Create the main title
    fig = Figlet(font="slant", width=80, justify="center")
    title_text = fig.renderText(title)

    # Apply gradient coloring - fixed parameter order
    colored_title = gradient_colorize(str(title_text), "CYAN", "BLUE")

    # Create banner structure
    banner = FigletString("═" * 80)
    result = banner + "\n" + colored_title

    # Add subtitle if provided
    if subtitle:
        small_fig = Figlet(font="small", width=80, justify="center")
        subtitle_text = small_fig.renderText(subtitle)
        colored_subtitle = pulse_colorize(str(subtitle_text), "LIGHT_CYAN")
        result += "\n" + colored_subtitle

    # Add bottom border
    result += "\n" + banner

    return result


def create_notice_box(text: str, style: str = "info") -> str:
    """
    Create a notice box with specified style.

    Args:
        text: The text to display
        style: Box style ('info', 'warning', 'error', 'success')

    Returns:
        Text in a decorated box
    """
    # Define style parameters
    styles = {
        "info": ("BLUE", "─ INFO ─"),
        "warning": ("YELLOW", "─ WARNING ─"),
        "error": ("RED", "─ ERROR ─"),
        "success": ("GREEN", "─ SUCCESS ─"),
    }

    color, title = styles.get(style.lower(), styles["info"])

    # Create the notice text
    fig = Figlet(font="small", width=70)
    text_art = fig.renderText(text)

    # Convert to FigletString if it's not already (ensuring we have the right type)
    figlet_text: Union[str, FigletString] = text_art

    # Use string methods instead of FigletString methods that may not be implemented
    if hasattr(figlet_text, "strip_surrounding_newlines"):
        figlet_text = figlet_text.strip_surrounding_newlines()
    else:
        # Fallback to standard string strip
        figlet_text = figlet_text.strip()

    # Create the box - using type guards for cleaner control flow
    box_text: str
    # Simple box drawing as fallback
    lines = str(figlet_text).splitlines()
    width = max(len(line) for line in lines) + 4
    box_text = "┌" + "─" * (width - 2) + "┐\n"
    box_text += "│ " + title.center(width - 4) + " │\n"
    box_text += "├" + "─" * (width - 2) + "┤\n"
    for line in lines:
        box_text += "│ " + line.ljust(width - 4) + " │\n"
    box_text += "└" + "─" * (width - 2) + "┘"

    # Apply color based on style - ensuring string type for pulse_colorize
    return pulse_colorize(box_text, color)


def create_animated_text(text: str, frames: int = 10) -> List[str]:
    """
    Create frames for animated text.

    Args:
        text: Text to animate
        frames: Number of animation frames

    Returns:
        List of frames for animation
    """
    fig = Figlet(font="standard", width=80, justify="center")
    base_text = fig.renderText(text)

    # Generate animation frames
    animation_frames: List[str] = []

    # Create different color effects for each frame - ensuring string type
    rainbow_frame = rainbow_colorize(str(base_text))
    animation_frames.append(rainbow_frame)

    # Create gradient frames with different color combinations
    color_pairs = [
        ("RED", "YELLOW"),
        ("YELLOW", "GREEN"),
        ("GREEN", "CYAN"),
        ("CYAN", "BLUE"),
        ("BLUE", "MAGENTA"),
        ("MAGENTA", "RED"),
    ]

    for start, end in color_pairs:
        gradient_frame = gradient_colorize(str(base_text), start, end)
        animation_frames.append(gradient_frame)

    flipped = base_text.flip()
    animation_frames.append(rainbow_colorize(str(flipped)))

    # Type guard ensures reverse method exists with correct signature
    reversed_text = base_text.reverse()
    animation_frames.append(rainbow_colorize(str(reversed_text)))

    return animation_frames


def demonstrate_usage() -> None:
    """Run a demonstration of advanced API usage."""
    print("\n=== FIGLET FORGE ADVANCED API DEMONSTRATION ===\n")

    # Create and display a title banner
    try:
        banner = create_title_banner("Figlet Forge", "Advanced Typography Engine")
        print(banner)
        print()
    except Exception as e:
        print(f"Banner creation failed: {e}")
        print("Falling back to simple text banner")
        print("=" * 50)
        print("FIGLET FORGE")
        print("Advanced Typography Engine")
        print("=" * 50)
        print()

    # Create and display notice boxes
    try:
        info_box = create_notice_box("This is an informational message", "info")
        print(info_box)
        print()

        warning_box = create_notice_box("Warning: Proceed with caution!", "warning")
        print(warning_box)
        print()
    except Exception as e:
        print(f"Notice box creation failed: {e}")

    # Demonstrate text animation
    print("Text Animation Demo:")
    try:
        frames = create_animated_text("Animate!")

        # Show animation frames (when running in terminal)
        if sys.stdout.isatty():
            try:
                for frame in frames * 2:  # Repeat animation twice
                    # Clear previous frame
                    print("\033[H\033[J")  # Clear screen
                    # Print new frame
                    print(frame)
                    # Wait briefly
                    time.sleep(0.5)
            except KeyboardInterrupt:
                print("\nAnimation stopped.")
        else:
            # Just show one frame if not in a terminal
            print(frames[0])
    except Exception as e:
        print(f"Animation creation failed: {e}")
        # Fallback to simple output
        simple_fig = Figlet()
        try:
            print(simple_fig.renderText("Animate!"))
        except Exception as specific_e:  # Replaced bare except with specific exception
            print(f"Simple render failed: {specific_e}")
            print("ANIMATE!")

    print("\n=== END OF DEMONSTRATION ===\n")


if __name__ == "__main__":
    demonstrate_usage()
