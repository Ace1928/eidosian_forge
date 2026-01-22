import functools
import re
from enum import (
from typing import (
from wcwidth import (  # type: ignore[import]
def style_aware_wcswidth(text: str) -> int:
    """
    Wrap wcswidth to make it compatible with strings that contain ANSI style sequences.
    This is intended for single line strings. If text contains a newline, this
    function will return -1. For multiline strings, call widest_line() instead.

    :param text: the string being measured
    :return: The width of the string when printed to the terminal if no errors occur.
             If text contains characters with no absolute width (i.e. tabs),
             then this function returns -1. Replace tabs with spaces before calling this.
    """
    return cast(int, wcswidth(strip_style(text)))