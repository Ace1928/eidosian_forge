import functools
import re
from enum import (
from typing import (
from wcwidth import (  # type: ignore[import]
def widest_line(text: str) -> int:
    """
    Return the width of the widest line in a multiline string. This wraps style_aware_wcswidth()
    so it handles ANSI style sequences and has the same restrictions on non-printable characters.

    :param text: the string being measured
    :return: The width of the string when printed to the terminal if no errors occur.
             If text contains characters with no absolute width (i.e. tabs),
             then this function returns -1. Replace tabs with spaces before calling this.
    """
    if not text:
        return 0
    lines_widths = [style_aware_wcswidth(line) for line in text.splitlines()]
    if -1 in lines_widths:
        return -1
    return max(lines_widths)