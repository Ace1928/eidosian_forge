from typing import List, Sequence, Tuple
import itertools
import numpy as np
import pytest
import sympy
import cirq
def test_circuit_structure():
    syc = FakeSycamoreGate()
    ops = cirq.decompose_cphase_into_two_fsim(cirq.CZ, fsim_gate=syc)
    num_interaction_moments = 0
    for op in ops:
        assert len(op.qubits) in (0, 1, 2)
        if len(op.qubits) == 2:
            num_interaction_moments += 1
            assert isinstance(op.gate, FakeSycamoreGate)
    assert num_interaction_moments == 2