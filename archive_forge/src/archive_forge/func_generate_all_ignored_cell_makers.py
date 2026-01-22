from typing import Iterator
from cirq.interop.quirk.cells.cell import CellMaker, CELL_SIZES
def generate_all_ignored_cell_makers() -> Iterator[CellMaker]:
    yield _ignored_gate('…')
    yield _ignored_gate('Bloch')
    yield from _ignored_family('Amps')
    yield from _ignored_family('Chance')
    yield from _ignored_family('Sample')
    yield from _ignored_family('Density')