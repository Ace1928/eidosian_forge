from typing import Any, Iterable, Iterator, List, Optional, TYPE_CHECKING
from cirq import ops, value
from cirq.interop.quirk.cells.cell import Cell, CellMaker
def generate_all_swap_cell_makers() -> Iterator[CellMaker]:
    yield CellMaker('Swap', 1, lambda args: SwapCell(args.qubits, []))