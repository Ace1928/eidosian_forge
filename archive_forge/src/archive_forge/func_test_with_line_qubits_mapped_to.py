import pytest
from cirq import quirk_url_to_circuit
from cirq.interop.quirk.cells.testing import assert_url_to_circuit_returns
from cirq.interop.quirk.cells.input_cells import SetDefaultInputCell
def test_with_line_qubits_mapped_to():
    cell = SetDefaultInputCell('a', 5)
    assert cell.with_line_qubits_mapped_to([]) is cell