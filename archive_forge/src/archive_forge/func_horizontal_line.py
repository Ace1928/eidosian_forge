from typing import (
import numpy as np
from cirq import value
from cirq.circuits._block_diagram_drawer import BlockDiagramDrawer
from cirq.circuits._box_drawing_character_data import (
def horizontal_line(self, y: Union[int, float], x1: Union[int, float], x2: Union[int, float], emphasize: bool=False, doubled: bool=False) -> None:
    """Adds a line from (x1, y) to (x2, y)."""
    x1, x2 = sorted([x1, x2])
    self.horizontal_lines.append(_HorizontalLine(y, x1, x2, emphasize, doubled))