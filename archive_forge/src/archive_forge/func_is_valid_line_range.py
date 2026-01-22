import difflib
from dataclasses import dataclass
from typing import Collection, Iterator, List, Sequence, Set, Tuple, Union
from black.nodes import (
from blib2to3.pgen2.token import ASYNC, NEWLINE
def is_valid_line_range(lines: Tuple[int, int]) -> bool:
    """Returns whether the line range is valid."""
    return not lines or lines[0] <= lines[1]