#!/usr/bin/env python3
"""
ASCII to SVG Conversion Example.

This script demonstrates how to convert ASCII art text to SVG format
using Figlet Forge's rendering capabilities.
"""

import sys
from pathlib import Path
from typing import (
    Any,
    Dict,
    Optional,
    Protocol,
    TypedDict,
    Union,
    cast,
    runtime_checkable,
)

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from figlet_forge import Figlet
from figlet_forge.core.figlet_string import FigletString
from figlet_forge.render.figlet_engine import RenderEngine


@runtime_checkable
class Borderable(Protocol):
    """Protocol for objects that can be bordered."""

    def border(self, *args: object, **kwargs: object) -> FigletString: ...


class SVGOptions(TypedDict, total=False):
    """
    TypedDict defining SVG rendering options for precise type checking.

    Using total=False allows for partial specification of options.
    """

    # Required fields
    font_family: str
    font_size: int
    foreground: str
    background: str
    padding: int

    # Optional enhanced fields
    x: int
    y: int
    line_height: Optional[int]
    width: Optional[int]
    height: Optional[int]
    additional_styles: Optional[Dict[str, str]]
    viewbox: Optional[str]
    accessibility: bool
    animation: Optional[Dict[str, Any]]
    svg_class: str
    line_class: str
    metadata: Optional[Dict[str, str]]
    responsive: bool


def safely_border_text(text: Union[FigletString, str]) -> Union[FigletString, str]:
    """
    Safely apply border to text if possible.

    This function specifically checks for the border method and handles
    the structural typing in a way that satisfies static analysis.

    Args:
        text: Text to border, could be FigletString or plain string

    Returns:
        Bordered text if possible, otherwise original text
    """
    # Explicit method attribute check rather than relying solely on Protocol matching
    if hasattr(text, "border") and callable(text.border):
        try:
            # We know border exists, but cast to satisfy type checker
            bordered = cast(Borderable, text).border()
            return bordered
        except Exception as e:
            print(f"Warning: Border method failed: {e}")
            return text
    return text


def main() -> None:
    """
    Demonstrate ASCII to SVG conversion.

    Takes command line input text or uses a default, renders it as ASCII art,
    adds a border, and converts to SVG format with specified styling options.

    Returns:
        None: Outputs SVG files to disk
    """
    try:
        # Get text from command line or use default
        text = sys.argv[1] if len(sys.argv) > 1 else "SVG Art!"

        # Create Figlet instance with a nice font
        fig = Figlet(font="slant", width=100, justify="center")

        # Render the text to ASCII art
        ascii_art = fig.renderText(text)

        # Add a border for better visual appearance using the safer function
        bordered_art = safely_border_text(ascii_art)

        if bordered_art == ascii_art:
            print("Note: Border function not available, using plain text.")

        # Print the ASCII art to console
        print("ASCII Art:")
        print(bordered_art)

        # Basic SVG options
        basic_svg_options: SVGOptions = {
            "font_family": "monospace, 'Courier New', Courier",
            "font_size": 16,
            "foreground": "#333",
            "background": "#f8f8f8",
            "padding": 20,
        }

        # Using RenderEngine with basic options - ensure we always pass string
        basic_svg_content: str = RenderEngine.to_svg(
            text=str(bordered_art), **basic_svg_options
        )

        # Enhanced SVG options with animations and metadata
        enhanced_svg_options: SVGOptions = {
            "font_family": "monospace, 'Courier New', Courier",
            "font_size": 16,
            "foreground": "#333",
            "background": "#f8f8f8",
            "padding": 20,
            "additional_styles": {"font-weight": "bold", "letter-spacing": "0.1em"},
            "animation": {"type": "fadeIn", "duration": 1.5},
            "metadata": {
                "generator": "Figlet Forge",
                "author": "Eidosian ASCII Art Generator",
                "content": text,
            },
            "responsive": True,
        }

        # Generate enhanced SVG - ensure we always pass string
        enhanced_svg_content: str = RenderEngine.to_svg(
            text=str(bordered_art), **enhanced_svg_options
        )

        # Save basic version to file with proper error handling
        basic_output_file = "figlet_svg_output.svg"
        try:
            with open(basic_output_file, "w") as f:
                f.write(basic_svg_content)
            print(f"\nBasic SVG version saved to '{basic_output_file}'")
        except OSError as e:
            print(f"Error saving basic SVG file: {e}")

        # Save enhanced version to file with proper error handling
        enhanced_output_file = "figlet_svg_output_enhanced.svg"
        try:
            with open(enhanced_output_file, "w") as f:
                f.write(enhanced_svg_content)
            print(f"Enhanced SVG version saved to '{enhanced_output_file}'")
            print("You can open them in a web browser or SVG editor.")
        except OSError as e:
            print(f"Error saving enhanced SVG file: {e}")
            return

    except Exception as e:
        print(f"Error in ASCII to SVG conversion: {e}")
        return


if __name__ == "__main__":
    main()
