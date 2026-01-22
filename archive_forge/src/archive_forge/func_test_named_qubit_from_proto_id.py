import pytest
import cirq
import cirq_google.api.v2 as v2
def test_named_qubit_from_proto_id():
    assert v2.named_qubit_from_proto_id('a') == cirq.NamedQubit('a')