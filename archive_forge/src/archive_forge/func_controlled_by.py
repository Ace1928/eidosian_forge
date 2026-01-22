from typing import (
from cirq import circuits
from cirq.interop.quirk.cells.cell import Cell
def controlled_by(self, qubit: 'cirq.Qid') -> 'CompositeCell':
    return self._transform_cells(lambda cell: cell.controlled_by(qubit))