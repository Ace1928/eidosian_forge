from typing import Optional, List, Iterator, Iterable, TYPE_CHECKING
from cirq.interop.quirk.cells.cell import Cell, CELL_SIZES, CellMaker
def generate_all_input_cell_makers() -> Iterator[CellMaker]:
    yield from _input_family('inputA', 'a')
    yield from _input_family('inputB', 'b')
    yield from _input_family('inputR', 'r')
    yield from _input_family('revinputA', 'a', rev=True)
    yield from _input_family('revinputB', 'b', rev=True)
    yield CellMaker('setA', 2, lambda args: SetDefaultInputCell('a', args.value))
    yield CellMaker('setB', 2, lambda args: SetDefaultInputCell('b', args.value))
    yield CellMaker('setR', 2, lambda args: SetDefaultInputCell('r', args.value))