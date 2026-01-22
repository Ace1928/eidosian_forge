import random
import pytest
import cirq
import cirq.testing as ct
import cirq.contrib.acquaintance as cca
def test_swap_permutation_gate():
    no_decomp = lambda op: isinstance(op, cirq.GateOperation) and op.gate == cirq.SWAP
    a, b = (cirq.NamedQubit('a'), cirq.NamedQubit('b'))
    gate = cca.SwapPermutationGate()
    assert gate.num_qubits() == 2
    circuit = cirq.Circuit(gate(a, b))
    circuit = cirq.expand_composite(circuit, no_decomp=no_decomp)
    assert tuple(circuit.all_operations()) == (cirq.SWAP(a, b),)
    no_decomp = lambda op: isinstance(op, cirq.GateOperation) and op.gate == cirq.CZ
    circuit = cirq.Circuit(cca.SwapPermutationGate(cirq.CZ)(a, b))
    circuit = cirq.expand_composite(circuit, no_decomp=no_decomp)
    assert tuple(circuit.all_operations()) == (cirq.CZ(a, b),)
    assert cirq.commutes(gate, cirq.ZZ)
    with pytest.raises(TypeError):
        cirq.commutes(gate, cirq.CCZ)