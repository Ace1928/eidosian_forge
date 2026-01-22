import functools
import itertools
from typing import Tuple, Type
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
from cirq.testing import EqualsTester, assert_allclose_up_to_global_phase
def test_clifford_gate_act_on_large_case():
    n, num_ops = (50, 1000)
    gate_candidate = [cirq.X, cirq.Y, cirq.Z, cirq.H, cirq.S, cirq.CNOT, cirq.CZ]
    for seed in range(10):
        prng = np.random.RandomState(seed)
        t1 = cirq.CliffordTableau(num_qubits=n)
        t2 = cirq.CliffordTableau(num_qubits=n)
        qubits = cirq.LineQubit.range(n)
        args1 = cirq.CliffordTableauSimulationState(tableau=t1, qubits=qubits, prng=prng)
        args2 = cirq.CliffordTableauSimulationState(tableau=t2, qubits=qubits, prng=prng)
        ops = []
        for _ in range(0, num_ops, 100):
            g = prng.randint(len(gate_candidate))
            indices = (prng.randint(n),) if g < 5 else prng.choice(n, 2, replace=False)
            cirq.act_on(gate_candidate[g], args1, qubits=[qubits[i] for i in indices], allow_decompose=False)
            ops.append(gate_candidate[g].on(*[qubits[i] for i in indices]))
        compiled_gate = cirq.CliffordGate.from_op_list(ops, qubits)
        cirq.act_on(compiled_gate, args2, qubits)
        assert args1.tableau == args2.tableau