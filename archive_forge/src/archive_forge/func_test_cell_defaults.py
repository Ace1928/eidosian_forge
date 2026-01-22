import pytest
import cirq
from cirq.interop.quirk.cells.cell import Cell, ExplicitOperationsCell
def test_cell_defaults():

    class BasicCell(Cell):

        def with_line_qubits_mapped_to(self, qubits):
            raise NotImplementedError()

        def gate_count(self) -> int:
            raise NotImplementedError()
    c = BasicCell()
    assert c.operations() == ()
    assert c.basis_change() == ()
    assert c.controlled_by(cirq.LineQubit(0)) is c
    x = []
    c.modify_column(x)
    assert x == []