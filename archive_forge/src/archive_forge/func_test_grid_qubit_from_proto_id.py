import pytest
import cirq
import cirq_google.api.v2 as v2
def test_grid_qubit_from_proto_id():
    assert v2.grid_qubit_from_proto_id('1_2') == cirq.GridQubit(1, 2)
    assert v2.grid_qubit_from_proto_id('10_2') == cirq.GridQubit(10, 2)
    assert v2.grid_qubit_from_proto_id('-1_2') == cirq.GridQubit(-1, 2)
    assert v2.grid_qubit_from_proto_id('q-1_2') == cirq.GridQubit(-1, 2)
    assert v2.grid_qubit_from_proto_id('q1_2') == cirq.GridQubit(1, 2)