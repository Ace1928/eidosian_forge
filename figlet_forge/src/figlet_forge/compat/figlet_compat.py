from eidosian_core import eidosian
"""
Compatibility layer for pyfiglet.

This module provides backward compatibility with the pyfiglet API,
allowing existing code to work with Figlet Forge with minimal changes.
"""

from typing import List

from ..core.exceptions import FontNotFound
from ..figlet import Figlet as FigletForge


class Figlet(FigletForge):
    """Compatibility class for pyfiglet.Figlet."""

    def __init__(
        self,
        font: str = "standard",
        direction: str = "auto",
        justify: str = "auto",
        width: int = 80,
        **kwargs,
    ):
        """
        Initialize the Figlet object.

        Args:
            font: Font name
            direction: Text direction ('auto', 'left-to-right', 'right-to-left')
            justify: Text justification ('auto', 'left', 'center', 'right')
            width: Maximum width of rendered output
            **kwargs: Additional keywords for backward compatibility
        """
        # Map old kwargs to new ones for compatibility
        unicode_aware = kwargs.pop("unicode", False)

        # Additional compatibility parameters
        self.r_to_l = kwargs.pop("r_to_l", False)
        if self.r_to_l:
            direction = "right-to-left"

        # Support older direction format styles
        direction = self._convert_direction(direction)

        # Handle font not found gracefully with better fallbacks
        try:
            super().__init__(
                font=font,
                direction=direction,
                justify=justify,
                width=width,
                unicode_aware=unicode_aware,
                **kwargs,
            )
        except FontNotFound as e:
            # Special handling for pyfiglet compatibility: certain fonts might use
            # different formats that need custom parsing
            if "German" in str(e) or "marker line" in str(e):
                # Try with the enhanced German format detection
                kwargs["enhanced_parser"] = True
                super().__init__(
                    font="standard",  # Fall back to standard when specified font fails
                    direction=direction,
                    justify=justify,
                    width=width,
                    unicode_aware=unicode_aware,
                    **kwargs,
                )
            else:
                raise

    def _convert_direction(self, direction: str) -> str:
        """
        Convert direction format between different styles for compatibility.

        Args:
            direction: Direction string in any supported format

        Returns:
            Direction string in standard format
        """
        direction = str(direction).lower()

        # Map all possible direction formats to standard ones
        if direction in ("auto", "default"):
            return "auto"
        elif direction in ("left-to-right", "left_to_right", "ltr"):
            return "left-to-right"
        elif direction in ("right-to-left", "right_to_left", "rtl"):
            return "right-to-left"

        # Default to auto for unknown values
        return "auto"

    @eidosian()
    def getDirection(self) -> str:
        """Get the current direction with underscore format for compatibility."""
        # Convert hyphenated format to underscore format for compatibility
        return self.direction.replace("-", "_")

    @eidosian()
    def getJustify(self) -> str:
        """Get the current justification."""
        return self.justify

    @eidosian()
    def getFonts(self) -> List[str]:
        """Get list of available fonts (compatibility method)."""
        return self.fonts

    @eidosian()
    def setFont(self, font: str) -> None:
        """
        Set the current font.

        Args:
            font: Name of the font to use
        """
        self.font = font
        self._font = self.get_figlet_font()

    @eidosian()
    def getRenderWidth(self, text: str) -> int:
        """
        Get the render width for the given text.

        Args:
            text: Text to calculate width for

        Returns:
            Width in characters
        """
        return self.get_render_width(text)


@eidosian()
def figlet_format(
    text: str, font: str = "standard", justify: str = "auto", width: int = 80, **kwargs
) -> str:
    """
    Return ASCII art text as a string.

    Args:
        text: The text to render
        font: The font to use
        justify: The justification ('auto', 'left', 'center', 'right')
        width: The maximum width
        **kwargs: Additional keywords for backward compatibility

    Returns:
        The rendered ASCII art text
    """
    try:
        fig = Figlet(font=font, justify=justify, width=width, **kwargs)
        return fig.renderText(text)
    except FontNotFound:
        # For pyfiglet compatibility, fall back to standard font on errors
        fig = Figlet(font="standard", justify=justify, width=width, **kwargs)
        return fig.renderText(text)


# Legacy pyfiglet alias used by compatibility tests and integrations.
renderText = figlet_format
