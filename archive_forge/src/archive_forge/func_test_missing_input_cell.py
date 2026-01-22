import pytest
from cirq import quirk_url_to_circuit
from cirq.interop.quirk.cells.testing import assert_url_to_circuit_returns
from cirq.interop.quirk.cells.input_cells import SetDefaultInputCell
def test_missing_input_cell():
    with pytest.raises(ValueError, match='Missing input'):
        _ = quirk_url_to_circuit('https://algassert.com/quirk#circuit={"cols":[["+=A2"]]}')