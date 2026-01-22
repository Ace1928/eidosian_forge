import itertools
import numpy as np
import pytest
import sympy
import cirq
def test_gate_equality():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(cirq.CSwapGate(), cirq.CSwapGate())
    eq.add_equality_group(cirq.CZPowGate(), cirq.CZPowGate())
    eq.add_equality_group(cirq.CCXPowGate(), cirq.CCXPowGate(), cirq.CCNotPowGate())
    eq.add_equality_group(cirq.CCZPowGate(), cirq.CCZPowGate())