from typing import (
import numpy as np
from cirq import value
from cirq.circuits._block_diagram_drawer import BlockDiagramDrawer
from cirq.circuits._box_drawing_character_data import (
def superimposed(self, other: 'cirq.TextDiagramDrawer') -> 'cirq.TextDiagramDrawer':
    return self.copy().superimpose(other)