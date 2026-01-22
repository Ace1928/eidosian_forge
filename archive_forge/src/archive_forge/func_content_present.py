from typing import (
import numpy as np
from cirq import value
from cirq.circuits._block_diagram_drawer import BlockDiagramDrawer
from cirq.circuits._box_drawing_character_data import (
def content_present(self, x: int, y: int) -> bool:
    """Determines if a line or printed text is at the given location."""
    if (x, y) in self.entries:
        return True
    if any((v.x == x and v.y1 < y < v.y2 for v in self.vertical_lines)):
        return True
    if any((line_y == y and x1 < x < x2 for line_y, x1, x2, _, _ in self.horizontal_lines)):
        return True
    return False