import sys
from typing import TYPE_CHECKING, Iterable, List
from ._loop import loop_last
def get_bottom(self, widths: Iterable[int]) -> str:
    """Get the bottom of a simple box.

        Args:
            widths (List[int]): Widths of columns.

        Returns:
            str: A string of box characters.
        """
    parts: List[str] = []
    append = parts.append
    append(self.bottom_left)
    for last, width in loop_last(widths):
        append(self.bottom * width)
        if not last:
            append(self.bottom_divider)
    append(self.bottom_right)
    return ''.join(parts)