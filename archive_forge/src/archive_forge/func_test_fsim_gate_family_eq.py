from typing import List, Union
import pytest
import sympy
import numpy as np
import cirq
import cirq_google
def test_fsim_gate_family_eq():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(cirq_google.FSimGateFamily(), cirq_google.FSimGateFamily(gate_types_to_check=ALL_POSSIBLE_FSIM_GATES), cirq_google.FSimGateFamily(gate_types_to_check=ALL_POSSIBLE_FSIM_GATES[::-1]))
    eq.add_equality_group(cirq_google.FSimGateFamily(allow_symbols=True), cirq_google.FSimGateFamily(gate_types_to_check=ALL_POSSIBLE_FSIM_GATES, allow_symbols=True), cirq_google.FSimGateFamily(gate_types_to_check=ALL_POSSIBLE_FSIM_GATES[::-1], allow_symbols=True))
    eq.add_equality_group(cirq_google.FSimGateFamily(gates_to_accept=[cirq_google.SYC, cirq.SQRT_ISWAP, cirq.SQRT_ISWAP, cirq.CZPowGate, cirq.PhasedISwapPowGate], allow_symbols=True), cirq_google.FSimGateFamily(gates_to_accept=[cirq.FSimGate(theta=np.pi / 2, phi=np.pi / 6), cirq.SQRT_ISWAP, cirq.CZPowGate, cirq.CZPowGate, cirq.PhasedISwapPowGate, cirq.PhasedISwapPowGate], gate_types_to_check=ALL_POSSIBLE_FSIM_GATES + [cirq.FSimGate], allow_symbols=True), cirq_google.FSimGateFamily(gates_to_accept=[cirq.FSimGate(theta=np.pi / 2, phi=np.pi / 6), cirq.SQRT_ISWAP, cirq.CZPowGate, cirq.PhasedISwapPowGate], gate_types_to_check=ALL_POSSIBLE_FSIM_GATES[::-1] + [cirq.FSimGate], allow_symbols=True))