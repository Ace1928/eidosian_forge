from typing import (
import numpy as np
from cirq import value
from cirq.circuits._block_diagram_drawer import BlockDiagramDrawer
from cirq.circuits._box_drawing_character_data import (
def transform_rows(column: Union[int, float], row: Union[int, float]) -> Tuple[Union[int, float], Union[int, float]]:
    return (column, row + (amount if row >= y else 0))