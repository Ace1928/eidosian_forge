import pytest
import cirq
import cirq_google.api.v2 as v2
def test_qubit_to_proto_id():
    assert v2.qubit_to_proto_id(cirq.GridQubit(1, 2)) == '1_2'
    assert v2.qubit_to_proto_id(cirq.GridQubit(10, 2)) == '10_2'
    assert v2.qubit_to_proto_id(cirq.GridQubit(-1, 2)) == '-1_2'
    assert v2.qubit_to_proto_id(cirq.LineQubit(1)) == '1'
    assert v2.qubit_to_proto_id(cirq.LineQubit(10)) == '10'
    assert v2.qubit_to_proto_id(cirq.LineQubit(-1)) == '-1'
    assert v2.qubit_to_proto_id(cirq.NamedQubit('named')) == 'named'