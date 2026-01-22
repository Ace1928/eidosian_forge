import random
import pytest
import cirq
import cirq.testing as ct
import cirq.contrib.acquaintance as cca
def test_uses_consistent_swap_gate():
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.Circuit([cca.SwapPermutationGate()(a, b), cca.SwapPermutationGate()(a, b)])
    assert cca.uses_consistent_swap_gate(circuit, cirq.SWAP)
    assert not cca.uses_consistent_swap_gate(circuit, cirq.CZ)
    circuit = cirq.Circuit([cca.SwapPermutationGate(cirq.CZ)(a, b), cca.SwapPermutationGate(cirq.CZ)(a, b)])
    assert cca.uses_consistent_swap_gate(circuit, cirq.CZ)
    assert not cca.uses_consistent_swap_gate(circuit, cirq.SWAP)
    circuit = cirq.Circuit([cca.SwapPermutationGate()(a, b), cca.SwapPermutationGate(cirq.CZ)(a, b)])
    assert not cca.uses_consistent_swap_gate(circuit, cirq.SWAP)
    assert not cca.uses_consistent_swap_gate(circuit, cirq.CZ)