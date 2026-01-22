from typing import Iterator
from cirq.interop.quirk.cells.cell import CellMaker, CELL_SIZES
def generate_all_unsupported_cell_makers() -> Iterator[CellMaker]:
    yield from _unsupported_gates('|0⟩⟨0|', '|1⟩⟨1|', '|+⟩⟨+|', '|-⟩⟨-|', '|X⟩⟨X|', '|/⟩⟨/|', '0', reason='postselection is not implemented in Cirq')
    yield from _unsupported_gates('__error__', '__unstable__UniversalNot', reason='unphysical operation.')
    yield from _unsupported_gates('XDetectControlReset', 'YDetectControlReset', 'ZDetectControlReset', reason='classical feedback is not implemented in Cirq.')
    yield from _unsupported_gates('X^⌈t⌉', 'X^⌈t-¼⌉', reason='discrete parameter')
    yield from _unsupported_family('Counting', reason='discrete parameter')
    yield from _unsupported_family('Uncounting', reason='discrete parameter')
    yield from _unsupported_family('>>t', reason='discrete parameter')
    yield from _unsupported_family('<<t', reason='discrete parameter')
    yield from _unsupported_family('add', reason='deprecated; use +=A instead')
    yield from _unsupported_family('sub', reason='deprecated; use -=A instead')
    yield from _unsupported_family('c+=ab', reason='deprecated; use +=AB instead')
    yield from _unsupported_family('c-=ab', reason='deprecated; use -=AB instead')