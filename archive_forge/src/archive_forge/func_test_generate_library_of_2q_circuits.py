import itertools
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, cast
import networkx as nx
import numpy as np
import pytest
import cirq
from cirq.experiments import (
from cirq.experiments.random_quantum_circuit_generation import (
def test_generate_library_of_2q_circuits():
    circuits = generate_library_of_2q_circuits(n_library_circuits=5, two_qubit_gate=cirq.CNOT, max_cycle_depth=13, random_state=9)
    assert len(circuits) == 5
    for circuit in circuits:
        assert len(circuit.all_qubits()) == 2
        assert sorted(circuit.all_qubits()) == cirq.LineQubit.range(2)
        for m1, m2 in zip(circuit.moments[::2], circuit.moments[1::2]):
            assert len(m1.operations) == 2
            assert len(m2.operations) == 1
            assert m2.operations[0].gate == cirq.CNOT