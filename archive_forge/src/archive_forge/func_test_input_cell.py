import pytest
from cirq import quirk_url_to_circuit
from cirq.interop.quirk.cells.testing import assert_url_to_circuit_returns
from cirq.interop.quirk.cells.input_cells import SetDefaultInputCell
def test_input_cell():
    assert_url_to_circuit_returns('{"cols":[["inputA4",1,1,1,"+=A4"]]}', maps={0: 0, 35: 37})
    assert_url_to_circuit_returns('{"cols":[["inputA3",1,1,"inputB3",1,1,"+=AB3"]]}', maps={0: 0, 153: 159, 72: 73, 288: 288})
    with pytest.raises(ValueError, match='Duplicate qids'):
        _ = quirk_url_to_circuit('https://algassert.com/quirk#circuit={"cols":[["+=A3","inputA3"]]}')