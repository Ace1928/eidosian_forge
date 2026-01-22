import contextlib
import re
import textwrap
from typing import Iterable, List, Tuple, Union
def strip_leading_blank_lines(text):
    """Return text with leading blank lines removed."""
    split = text.splitlines()
    found = next((index for index, line in enumerate(split) if line.strip()), 0)
    return '\n'.join(split[found:])