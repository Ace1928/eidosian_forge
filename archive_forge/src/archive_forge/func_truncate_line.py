import argparse
import collections
import functools
import glob
import inspect
import itertools
import os
import re
import subprocess
import sys
import threading
import unicodedata
from enum import (
from typing import (
from . import (
from .argparse_custom import (
def truncate_line(line: str, max_width: int, *, tab_width: int=4) -> str:
    """
    Truncate a single line to fit within a given display width. Any portion of the string that is truncated
    is replaced by a '…' character. Supports characters with display widths greater than 1. ANSI style sequences
    do not count toward the display width.

    If there are ANSI style sequences in the string after where truncation occurs, this function will append them
    to the returned string.

    This is done to prevent issues caused in cases like: truncate_line(Fg.BLUE + hello + Fg.RESET, 3)
    In this case, "hello" would be truncated before Fg.RESET resets the color from blue. Appending the remaining style
    sequences makes sure the style is in the same state had the entire string been printed. align_text() relies on this
    behavior when preserving style over multiple lines.

    :param line: text to truncate
    :param max_width: the maximum display width the resulting string is allowed to have
    :param tab_width: any tabs in the text will be replaced with this many spaces
    :return: line that has a display width less than or equal to width
    :raises: ValueError if text contains an unprintable character like a newline
    :raises: ValueError if max_width is less than 1
    """
    import io
    from . import ansi
    line = line.replace('\t', ' ' * tab_width)
    if ansi.style_aware_wcswidth(line) == -1:
        raise ValueError('text contains an unprintable character')
    if max_width < 1:
        raise ValueError('max_width must be at least 1')
    if ansi.style_aware_wcswidth(line) <= max_width:
        return line
    styles_dict = get_styles_dict(line)
    done = False
    index = 0
    total_width = 0
    truncated_buf = io.StringIO()
    while not done:
        if index in styles_dict:
            truncated_buf.write(styles_dict[index])
            style_len = len(styles_dict[index])
            styles_dict.pop(index)
            index += style_len
            continue
        char = line[index]
        char_width = ansi.style_aware_wcswidth(char)
        if char_width + total_width >= max_width:
            char = constants.HORIZONTAL_ELLIPSIS
            char_width = ansi.style_aware_wcswidth(char)
            done = True
        total_width += char_width
        truncated_buf.write(char)
        index += 1
    remaining_styles = _remove_overridden_styles(list(styles_dict.values()))
    truncated_buf.write(''.join(remaining_styles))
    return truncated_buf.getvalue()