from typing import (
from cirq import circuits
from cirq.interop.quirk.cells.cell import Cell
def gate_count(self) -> int:
    return self._gate_count