import re
from functools import lru_cache
from typing import Callable, List
from ._cell_widths import CELL_WIDTHS
def set_cell_size(text: str, total: int) -> str:
    """Set the length of a string to fit within given number of cells."""
    if _is_single_cell_widths(text):
        size = len(text)
        if size < total:
            return text + ' ' * (total - size)
        return text[:total]
    if total <= 0:
        return ''
    cell_size = cell_len(text)
    if cell_size == total:
        return text
    if cell_size < total:
        return text + ' ' * (total - cell_size)
    start = 0
    end = len(text)
    while True:
        pos = (start + end) // 2
        before = text[:pos + 1]
        before_len = cell_len(before)
        if before_len == total + 1 and cell_len(before[-1]) == 2:
            return before[:-1] + ' '
        if before_len == total:
            return before
        if before_len > total:
            end = pos
        else:
            start = pos