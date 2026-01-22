from typing import Dict, List, Optional, Tuple
import collections
from cirq.circuits._box_drawing_character_data import box_draw_character, BoxDrawCharacterSet
def set_row_min_height(self, y: int, min_height: int):
    """Sets a minimum height for blocks in the row with coordinate y."""
    if y < 0:
        raise IndexError('y < 0')
    self._min_heights[y] = min_height