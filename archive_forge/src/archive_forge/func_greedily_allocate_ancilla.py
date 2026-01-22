import itertools
import cirq
import cirq_ft
from cirq_ft import infra
import numpy as np
import pytest
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
def greedily_allocate_ancilla(circuit: cirq.AbstractCircuit) -> cirq.Circuit:
    greedy_mm = cirq.GreedyQubitManager(prefix='ancilla', maximize_reuse=True)
    circuit = cirq.map_clean_and_borrowable_qubits(circuit, qm=greedy_mm)
    assert len(circuit.all_qubits()) < 30
    return circuit.unfreeze()