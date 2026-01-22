from typing import Any, Iterable, Iterator, List, Optional, TYPE_CHECKING, Union
from cirq import ops, value
from cirq.interop.quirk.cells.cell import Cell, CellMaker
def generate_all_control_cell_makers() -> Iterator[CellMaker]:
    yield _reg_control('•', basis_change=None)
    yield _reg_control('◦', basis_change=ops.X)
    yield _reg_control('⊕', basis_change=ops.Y ** 0.5)
    yield _reg_control('⊖', basis_change=ops.Y ** (-0.5))
    yield _reg_control('⊗', basis_change=ops.X ** (-0.5))
    yield _reg_control('(/)', basis_change=ops.X ** 0.5)
    yield _reg_parity_control('xpar', basis_change=ops.Y ** 0.5)
    yield _reg_parity_control('ypar', basis_change=ops.X ** (-0.5))
    yield _reg_parity_control('zpar', basis_change=None)