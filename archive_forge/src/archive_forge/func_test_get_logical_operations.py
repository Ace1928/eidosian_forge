import random
import pytest
import cirq
import cirq.testing as ct
import cirq.contrib.acquaintance as cca
def test_get_logical_operations():
    a, b, c, d = qubits = cirq.LineQubit.range(4)
    mapping = dict(zip(qubits, qubits))
    operations = [cirq.ZZ(a, b), cca.SwapPermutationGate()(b, c), cirq.SWAP(a, b), cca.SwapPermutationGate()(c, d), cca.SwapPermutationGate()(b, c), cirq.ZZ(a, b)]
    assert list(cca.get_logical_operations(operations, mapping)) == [cirq.ZZ(a, b), cirq.SWAP(a, c), cirq.ZZ(a, d)]