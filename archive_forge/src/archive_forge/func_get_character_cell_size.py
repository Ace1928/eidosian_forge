import re
from functools import lru_cache
from typing import Callable, List
from ._cell_widths import CELL_WIDTHS
@lru_cache(maxsize=4096)
def get_character_cell_size(character: str) -> int:
    """Get the cell size of a character.

    Args:
        character (str): A single character.

    Returns:
        int: Number of cells (0, 1 or 2) occupied by that character.
    """
    return _get_codepoint_cell_size(ord(character))