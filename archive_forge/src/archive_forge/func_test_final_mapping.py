import itertools
import pytest
import cirq
import cirq.contrib.acquaintance as cca
import cirq.contrib.routing as ccr
def test_final_mapping():
    n_qubits = 10
    qubits = cirq.LineQubit.range(n_qubits)
    initial_mapping = dict(zip(qubits, qubits))
    expected_final_mapping = dict(zip(qubits, reversed(qubits)))
    SWAP = cca.SwapPermutationGate()
    circuit = cirq.Circuit((cirq.Moment((SWAP(*qubits[i:i + 2]) for i in range(l % 2, n_qubits - 1, 2))) for l in range(n_qubits)))
    swap_network = ccr.SwapNetwork(circuit, initial_mapping)
    assert swap_network.final_mapping() == expected_final_mapping