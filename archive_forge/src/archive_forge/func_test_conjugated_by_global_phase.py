import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_conjugated_by_global_phase():
    a = cirq.LineQubit(0)
    assert cirq.X(a).conjugated_by(cirq.global_phase_operation(1j)) == cirq.X(a)
    assert cirq.Z(a).conjugated_by(cirq.global_phase_operation(np.exp(1.1j))) == cirq.Z(a)

    class DecomposeGlobal(cirq.Gate):

        def num_qubits(self):
            return 1

        def _decompose_(self, qubits):
            yield cirq.global_phase_operation(1j)
    assert cirq.X(a).conjugated_by(DecomposeGlobal().on(a)) == cirq.X(a)