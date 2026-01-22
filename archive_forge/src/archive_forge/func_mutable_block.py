from typing import Dict, List, Optional, Tuple
import collections
from cirq.circuits._box_drawing_character_data import box_draw_character, BoxDrawCharacterSet
def mutable_block(self, x: int, y: int) -> Block:
    """Returns the block at (x, y) so it can be edited."""
    if x < 0 or y < 0:
        raise IndexError('x < 0 or y < 0')
    return self._blocks[x, y]