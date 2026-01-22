import itertools
import random
from typing import Any
import numpy as np
import pytest
import sympy
import cirq
from cirq.transformers.analytical_decompositions.two_qubit_to_fsim import (
def test_decompose_two_qubit_interaction_into_four_fsim_gates():
    iswap = cirq.FSimGate(theta=np.pi / 2, phi=0)
    c = cirq.decompose_two_qubit_interaction_into_four_fsim_gates(np.eye(4), fsim_gate=iswap)
    assert set(c.all_qubits()) == set(cirq.LineQubit.range(2))
    c = cirq.decompose_two_qubit_interaction_into_four_fsim_gates(cirq.CZ, fsim_gate=iswap)
    assert set(c.all_qubits()) == set(cirq.LineQubit.range(2))
    c = cirq.decompose_two_qubit_interaction_into_four_fsim_gates(cirq.CZ(*cirq.LineQubit.range(20, 22)), fsim_gate=iswap)
    assert set(c.all_qubits()) == set(cirq.LineQubit.range(20, 22))
    c = cirq.decompose_two_qubit_interaction_into_four_fsim_gates(np.eye(4), fsim_gate=iswap, qubits=cirq.LineQubit.range(10, 12))
    assert set(c.all_qubits()) == set(cirq.LineQubit.range(10, 12))
    c = cirq.decompose_two_qubit_interaction_into_four_fsim_gates(cirq.CZ(*cirq.LineQubit.range(20, 22)), fsim_gate=iswap, qubits=cirq.LineQubit.range(10, 12))
    assert set(c.all_qubits()) == set(cirq.LineQubit.range(10, 12))