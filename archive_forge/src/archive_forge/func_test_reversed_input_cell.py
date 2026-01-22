import pytest
from cirq import quirk_url_to_circuit
from cirq.interop.quirk.cells.testing import assert_url_to_circuit_returns
from cirq.interop.quirk.cells.input_cells import SetDefaultInputCell
def test_reversed_input_cell():
    assert_url_to_circuit_returns('{"cols":[["revinputA4",1,1,1,"+=A4"]]}', maps={0: 0, 35: 39, 19: 27})
    assert_url_to_circuit_returns('{"cols":[["revinputA3",1,1,"revinputB3",1,1,"+=AB3"]]}', maps={0: 0, 177: 183, 72: 72, 288: 289})
    with pytest.raises(ValueError, match='Duplicate qids'):
        _ = quirk_url_to_circuit('https://algassert.com/quirk#circuit={"cols":[["+=A3","revinputA3"]]}')