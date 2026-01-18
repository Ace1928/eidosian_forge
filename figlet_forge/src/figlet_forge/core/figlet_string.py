"""
Figlet string class for ASCII art text manipulation.

This module provides the FigletString class which extends the built-in
string class with methods specifically designed for manipulating ASCII art,
enabling transformations like mirroring, flipping, and adding borders.

Following Eidosian principles, it implements precise transformations
while maintaining structural elegance.
"""

from typing import List, Optional, Tuple, TypeVar, Union, cast

T = TypeVar("T", bound="FigletString")


class FigletString(str):
    """
    Rendered figlet font string with transformation capabilities.

    A specialized string class for ASCII art text that provides
    transformation operations specifically designed for FIGlet output.
    Maintains the structural integrity of ASCII art during manipulation.

    FigletString inherits from str but adds methods that are aware of the
    multi-line nature of ASCII art, preserving its structure during transformations.
    """

    # Translation maps for transforming ASCII art
    # These maps are used by the reverse() and flip() methods to maintain
    # visual integrity when transforming ASCII/Unicode art

    # Map for horizontal mirroring (left-to-right reversal)
    # Each character is mapped to its mirror counterpart across vertical axis
    HORIZONTAL_MIRROR_MAP = {
        "(": ")",
        ")": "(",
        "[": "]",
        "]": "[",
        "{": "}",
        "}": "{",
        "<": ">",
        ">": "<",
        "/": "\\",
        "\\": "/",
        "⟨": "⟩",
        "⟩": "⟨",
        "⟮": "⟯",
        "⟯": "⟮",
        "⦃": "⦄",
        "⦄": "⦃",
        "⦇": "⦈",
        "⦈": "⦇",
        "⟦": "⟧",
        "⟧": "⟦",
        "⟨": "⟩",
        "⟩": "⟨",
        "⟪": "⟫",
        "⟫": "⟪",
        "⌈": "⌉",
        "⌉": "⌈",
        "⌊": "⌋",
        "⌋": "⌊",
        "⦑": "⦒",
        "⦒": "⦑",
        "⧼": "⧽",
        "⧽": "⧼",
    }

    # Map for vertical flipping (top-to-bottom reversal)
    # Each character is mapped to its flip counterpart across horizontal axis
    VERTICAL_FLIP_MAP = {
        ".": "˙",
        ",": "'",
        "'": ",",
        "`": ",",
        ";": "؛",
        "!": "¡",
        "?": "¿",
        "(": ")",
        ")": "(",
        "[": "]",
        "]": "[",
        "{": "}",
        "}": "{",
        "<": ">",
        ">": "<",
        "^": "v",
        "v": "^",
        "b": "q",
        "d": "p",
        "p": "d",
        "q": "b",
        "u": "n",
        "n": "u",
        "A": "V",
        "V": "A",
        "M": "W",
        "W": "M",
        "⌄": "∧",
        "∧": "⌄",
        "∨": "∧",
        "∧": "∨",
    }

    # Border styles for the border() method
    BORDER_STYLES = {
        # Format: (top-left, top, top-right, left, right, bottom-left, bottom, bottom-right)
        "single": ("┌", "─", "┐", "│", "│", "└", "─", "┘"),
        "double": ("╔", "═", "╗", "║", "║", "╚", "═", "╝"),
        "rounded": ("╭", "─", "╮", "│", "│", "╰", "─", "╯"),
        "bold": ("┏", "━", "┓", "┃", "┃", "┗", "━", "┛"),
        "ascii": ("+", "-", "+", "|", "|", "+", "-", "+"),
        "shadow": ("┌", "─", "┐", "│", "│", "└", "─", "┘", "█", "▀", "▄"),
    }

    def __new__(cls, content: str, *args, **kwargs) -> "FigletString":
        """
        Create a new FigletString instance.

        Args:
            content: The string content
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            New FigletString instance
        """
        return super(FigletString, cls).__new__(cls, content)

    @property
    def dimensions(self) -> Tuple[int, int]:
        """
        Get the dimensions (width, height) of the ASCII art.

        The width is the maximum length of any line, and
        the height is the number of lines.

        Returns:
            Tuple of (width, height)
        """
        lines = self.splitlines()
        width = max((len(line) for line in lines), default=0)
        height = len(lines)
        return (width, height)

    def get_size(self) -> Tuple[int, int]:
        """
        Get the size of the FigletString as (width, height).

        Returns:
            Tuple containing width and height
        """
        return self.dimensions

    def splitlines(self) -> List[str]:
        """
        Split the FigletString into lines, preserving the str type.

        Returns:
            List of string objects, one per line
        """
        # Return plain strings to avoid recursion issues
        return super().splitlines()

    def _to_figlet_lines(self) -> List["FigletString"]:
        """
        Split into lines, preserving the FigletString type.

        Returns:
            List of FigletString objects, one per line
        """
        lines = super().splitlines()
        return [FigletString(line) for line in lines]

    def reverse(self: T) -> T:
        """
        Reverse the FigletString horizontally (mirror effect).

        This maintains the visual integrity of ASCII art by translating
        characters that have directional meaning using the HORIZONTAL_MIRROR_MAP.

        Returns:
            New FigletString with reversed content
        """
        lines = super().splitlines()
        reversed_lines = []

        for line in lines:
            # Reverse the line and translate characters using the mirror map
            reversed_line = ""
            for char in reversed(line):
                reversed_line += self.HORIZONTAL_MIRROR_MAP.get(char, char)

            reversed_lines.append(reversed_line)

        return cast(T, FigletString("\n".join(reversed_lines)))

    def flip(self: T) -> T:
        """
        Flip the FigletString vertically (upside down).

        This maintains the visual integrity of ASCII art by translating
        characters that have directional meaning using the VERTICAL_FLIP_MAP.

        Returns:
            New FigletString with flipped content
        """
        lines = super().splitlines()
        flipped_lines = []

        for line in reversed(lines):
            # Translate characters using the flip map
            flipped_line = ""
            for char in line:
                flipped_line += self.VERTICAL_FLIP_MAP.get(char, char)

            flipped_lines.append(flipped_line)

        return cast(T, FigletString("\n".join(flipped_lines)))

    def center(self: T, width: Optional[int] = None, fillchar: str = " ") -> T:
        """
        Center the FigletString within a field of specified width.

        If width is not provided, uses the maximum line length plus 10%.

        Args:
            width: Field width (default: max line length + 10%)
            fillchar: Character to use for padding

        Returns:
            New FigletString with centered content
        """
        lines = super().splitlines()

        # Calculate width if not provided
        if width is None:
            max_len = max(len(line) for line in lines)
            width = int(max_len * 1.1)  # Add 10% padding

        # Center each line
        centered_lines = [line.center(width, fillchar) for line in lines]

        return cast(T, FigletString("\n".join(centered_lines)))

    def ljust(self: T, width: Optional[int] = None, fillchar: str = " ") -> T:
        """
        Left-justify the FigletString within a field of specified width.

        If width is not provided, uses the maximum line length plus 10%.

        Args:
            width: Field width (default: max line length + 10%)
            fillchar: Character to use for padding

        Returns:
            New FigletString with left-justified content
        """
        lines = super().splitlines()

        # Calculate width if not provided
        if width is None:
            max_len = max(len(line) for line in lines)
            width = int(max_len * 1.1)  # Add 10% padding

        # Left-justify each line
        justified_lines = [line.ljust(width, fillchar) for line in lines]

        return cast(T, FigletString("\n".join(justified_lines)))

    def rjust(self: T, width: Optional[int] = None, fillchar: str = " ") -> T:
        """
        Right-justify the FigletString within a field of specified width.

        If width is not provided, uses the maximum line length plus 10%.

        Args:
            width: Field width (default: max line length + 10%)
            fillchar: Character to use for padding

        Returns:
            New FigletString with right-justified content
        """
        lines = super().splitlines()

        # Calculate width if not provided
        if width is None:
            max_len = max(len(line) for line in lines)
            width = int(max_len * 1.1)  # Add 10% padding

        # Right-justify each line
        justified_lines = [line.rjust(width, fillchar) for line in lines]

        return cast(T, FigletString("\n".join(justified_lines)))

    def border(self: T, style: str = "single", padding: int = 1) -> T:
        """
        Add a border around the FigletString.

        Args:
            style: Border style (single, double, rounded, bold, ascii, shadow)
            padding: Amount of padding inside the border

        Returns:
            New FigletString with border
        """
        if style not in self.BORDER_STYLES:
            raise ValueError(
                f"Unknown border style: {style}. Available styles: {', '.join(self.BORDER_STYLES.keys())}"
            )

        lines = super().splitlines()

        # Get the maximum width of the content
        max_width = max(len(line) for line in lines) if lines else 0

        # Get border characters
        border = self.BORDER_STYLES[style]

        # Create the bordered output
        result = []

        # Add top border
        result.append(f"{border[0]}{border[1] * (max_width + 2 * padding)}{border[2]}")

        # Add padding rows if needed
        if padding > 1:
            for _ in range(padding - 1):
                result.append(
                    f"{border[3]}{' ' * (max_width + 2 * padding)}{border[4]}"
                )

        # Add content with left and right borders
        for line in lines:
            result.append(
                f"{border[3]}{' ' * padding}{line.ljust(max_width)}{' ' * padding}{border[4]}"
            )

        # Add padding rows if needed
        if padding > 1:
            for _ in range(padding - 1):
                result.append(
                    f"{border[3]}{' ' * (max_width + 2 * padding)}{border[4]}"
                )

        # Add bottom border
        result.append(f"{border[5]}{border[6] * (max_width + 2 * padding)}{border[7]}")

        # Add shadow if that style is selected
        if style == "shadow" and len(border) > 8:
            shadow_lines = []
            # First line has shadow on the right side
            shadow_lines.append(result[0] + border[8])

            # Middle lines have shadow on the right side
            for i in range(1, len(result) - 1):
                shadow_lines.append(result[i] + border[8])

            # Last line has shadow on the bottom and bottom-right
            last_line = result[-1]
            shadow_lines.append(last_line + border[8])
            shadow_lines.append(" " + border[9] * (len(last_line) - 1) + border[10])

            result = shadow_lines

        return cast(T, FigletString("\n".join(result)))

    def shadow(self: T) -> T:
        """
        Add a shadow effect to the FigletString.

        This is a convenience method that calls border() with style="shadow".

        Returns:
            New FigletString with shadow effect
        """
        return self.border(style="shadow")

    def overlay(
        self: T, other: Union[str, "FigletString"], x: int = 0, y: int = 0
    ) -> T:
        """
        Overlay another FigletString or string on top of this one at position (x,y).

        Args:
            other: FigletString or string to overlay
            x: Horizontal position (0 = left)
            y: Vertical position (0 = top)

        Returns:
            New FigletString with overlay
        """
        if not isinstance(other, (str, FigletString)):
            raise TypeError(f"Expected str or FigletString, got {type(other).__name__}")

        # Convert to FigletString if necessary
        if not isinstance(other, FigletString):
            other = FigletString(other)

        self_lines = super().splitlines()
        other_lines = other.splitlines()

        result = list(self_lines)  # Create a copy of self_lines

        # Ensure result has enough lines for the overlay at position y
        while len(result) <= y + len(other_lines) - 1:
            result.append("")

        # Apply overlay
        for i, other_line in enumerate(other_lines):
            # Skip if outside the vertical bounds
            if y + i < 0:
                continue

            # Get the target line or empty string if it's outside the bounds
            if y + i >= len(result):
                target_line = ""
            else:
                target_line = result[y + i]

            # Create the overlaid line
            if x < 0:
                # Overlay is to the left of the origin
                overlaid = other_line[abs(x) :] + target_line
            else:
                # Ensure target_line is long enough
                if len(target_line) < x:
                    target_line = target_line + " " * (x - len(target_line))

                # Overlay at position x
                overlaid = (
                    target_line[:x] + other_line + target_line[x + len(other_line) :]
                )

            result[y + i] = overlaid

        return cast(T, FigletString("\n".join(result)))

    def strip_surrounding_newlines(self: T) -> T:
        """
        Remove leading and trailing newlines.

        Returns:
            New FigletString with surrounding newlines removed
        """
        return cast(T, FigletString(self.strip("\n")))

    def scale(self: T, horizontal: float = 1.0, vertical: float = 1.0) -> T:
        """
        Scale the FigletString horizontally and/or vertically.

        Args:
            horizontal: Horizontal scale factor (1.0 = no change)
            vertical: Vertical scale factor (1.0 = no change)

        Returns:
            New FigletString with scaled content
        """
        if horizontal <= 0 or vertical <= 0:
            raise ValueError("Scale factors must be positive")

        lines = super().splitlines()
        result = []

        # Scale vertically by duplicating lines
        for line in lines:
            # Scale horizontally by duplicating characters
            scaled_line = "".join(char * int(horizontal) for char in line)

            # Add fractional part if needed
            if horizontal % 1 > 0:
                fraction = horizontal % 1
                for i in range(len(line)):
                    if i / len(line) < fraction:
                        scaled_line += line[i]

            # Duplicate the scaled line according to vertical scale
            for _ in range(int(vertical)):
                result.append(scaled_line)

        # Add fractional vertical scaling if needed
        if vertical % 1 > 0:
            fraction = vertical % 1
            for i in range(len(lines)):
                if i / len(lines) < fraction:
                    result.append(lines[i])

        return cast(T, FigletString("\n".join(result)))

    def crop(
        self: T,
        left: int = 0,
        top: int = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> T:
        """
        Crop the FigletString to the specified region.

        Args:
            left: Left edge of crop region
            top: Top edge of crop region
            width: Width of crop region (None = to the end)
            height: Height of crop region (None = to the end)

        Returns:
            New FigletString with cropped content
        """
        lines = super().splitlines()
        result = []

        # Calculate actual width and height
        full_width = max(len(line) for line in lines) if lines else 0
        full_height = len(lines)

        # Default crop dimensions to full size if not specified
        if width is None:
            width = full_width - left
        if height is None:
            height = full_height - top

        # Ensure crop region is valid
        if left < 0 or top < 0 or width <= 0 or height <= 0:
            raise ValueError("Crop region must have positive dimensions")

        # Crop vertically
        crop_lines = lines[top : top + height] if top < len(lines) else []

        # Crop horizontally
        for line in crop_lines:
            if left < len(line):
                result.append(line[left : left + width])
            else:
                result.append("")

        return cast(T, FigletString("\n".join(result)))

    def rotate_90_clockwise(self: T) -> T:
        """
        Rotate the FigletString 90 degrees clockwise.

        Returns:
            New FigletString with rotated content
        """
        lines = super().splitlines()
        if not lines:
            return cast(T, FigletString(""))

        # Determine dimensions
        height = len(lines)
        width = max(len(line) for line in lines)

        # Pad lines to equal width
        padded_lines = [line.ljust(width) for line in lines]

        # Create rotated result
        rotated = []
        for i in range(width):
            new_line = ""
            for j in range(height - 1, -1, -1):
                if i < len(padded_lines[j]):
                    new_line += padded_lines[j][i]
                else:
                    new_line += " "
            rotated.append(new_line)

        return cast(T, FigletString("\n".join(rotated)))

    def rotate_90_counterclockwise(self: T) -> T:
        """
        Rotate the FigletString 90 degrees counterclockwise.

        Returns:
            New FigletString with rotated content
        """
        lines = super().splitlines()
        if not lines:
            return cast(T, FigletString(""))

        # Determine dimensions
        height = len(lines)
        width = max(len(line) for line in lines)

        # Pad lines to equal width
        padded_lines = [line.ljust(width) for line in lines]

        # Create rotated result
        rotated = []
        for i in range(width - 1, -1, -1):
            new_line = ""
            for j in range(height):
                if i < len(padded_lines[j]):
                    new_line += padded_lines[j][i]
                else:
                    new_line += " "
            rotated.append(new_line)

        return cast(T, FigletString("\n".join(rotated)))

    def __add__(self, other: object) -> "FigletString":
        """
        Concatenate with another string or FigletString.

        Args:
            other: String or FigletString to concatenate with

        Returns:
            New FigletString with concatenated content

        Raises:
            TypeError: If other is not a string or FigletString
        """
        if isinstance(other, (str, FigletString)):
            return FigletString(super().__add__(other))
        return NotImplemented

    def __repr__(self) -> str:
        """
        Return a string representation of the FigletString.

        Returns:
            String representation
        """
        return f"FigletString({super().__repr__()})"
