import numpy as np
import pytest
import cirq
from cirq.interop.quirk.cells.testing import assert_url_to_circuit_returns
from cirq import quirk_url_to_circuit
def test_input_rotation_cell_with_qubits():
    a, b, c, d, e = cirq.LineQubit.range(5)
    x, y, z, t, w = cirq.LineQubit.range(10, 15)
    cell = cirq.interop.quirk.cells.input_rotation_cells.InputRotationCell(identifier='test', register=[a, b, c], base_operation=cirq.X(d).controlled_by(e), exponent_sign=-1)
    assert cell.with_line_qubits_mapped_to([x, y, z, t, w]) == cirq.interop.quirk.cells.input_rotation_cells.InputRotationCell(identifier='test', register=[x, y, z], base_operation=cirq.X(t).controlled_by(w), exponent_sign=-1)