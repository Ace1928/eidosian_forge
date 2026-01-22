from typing import (
import numpy as np
from cirq import value
from cirq.circuits._block_diagram_drawer import BlockDiagramDrawer
from cirq.circuits._box_drawing_character_data import (
def vertical_line(self, x: Union[int, float], y1: Union[int, float], y2: Union[int, float], emphasize: bool=False, doubled: bool=False) -> None:
    """Adds a line from (x, y1) to (x, y2)."""
    y1, y2 = sorted([y1, y2])
    self.vertical_lines.append(_VerticalLine(x, y1, y2, emphasize, doubled))