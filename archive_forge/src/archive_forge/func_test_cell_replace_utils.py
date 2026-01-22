import pytest
import cirq
from cirq.interop.quirk.cells.cell import Cell, ExplicitOperationsCell
def test_cell_replace_utils():
    a, b, c = cirq.NamedQubit.range(3, prefix='q')
    assert Cell._replace_qubit(cirq.LineQubit(1), [a, b, c]) == b
    with pytest.raises(ValueError, match='only map from line qubits'):
        _ = Cell._replace_qubit(cirq.GridQubit(0, 0), [a, b, c])
    with pytest.raises(ValueError, match='not in range'):
        _ = Cell._replace_qubit(cirq.LineQubit(-1), [a, b, c])
    with pytest.raises(ValueError, match='not in range'):
        _ = Cell._replace_qubit(cirq.LineQubit(999), [a, b, c])