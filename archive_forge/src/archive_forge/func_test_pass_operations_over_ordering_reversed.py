import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_pass_operations_over_ordering_reversed():

    class OrderSensitiveGate(cirq.Gate):

        def num_qubits(self):
            return 2

        def _decompose_(self, qubits):
            return [cirq.Y(qubits[0]) ** (-0.5), cirq.CNOT(*qubits)]
    a, b = cirq.LineQubit.range(2)
    inp = cirq.X(a) * cirq.Z(b)
    out1 = inp.pass_operations_over([OrderSensitiveGate().on(a, b)], after_to_before=True)
    out2 = inp.pass_operations_over([cirq.Y(a) ** (-0.5), cirq.CNOT(a, b)], after_to_before=True)
    out3 = inp.pass_operations_over([cirq.Y(a) ** (-0.5)], after_to_before=True).pass_operations_over([cirq.CNOT(a, b)], after_to_before=True)
    assert out1 == out2 == out3 == cirq.Z(b)