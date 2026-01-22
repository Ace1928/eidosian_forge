import re
from functools import lru_cache
from typing import Callable, List
from ._cell_widths import CELL_WIDTHS
def cell_len(text: str, _cell_len: Callable[[str], int]=cached_cell_len) -> int:
    """Get the number of cells required to display text.

    Args:
        text (str): Text to display.

    Returns:
        int: Get the number of cells required to display text.
    """
    if len(text) < 512:
        return _cell_len(text)
    _get_size = get_character_cell_size
    total_size = sum((_get_size(character) for character in text))
    return total_size