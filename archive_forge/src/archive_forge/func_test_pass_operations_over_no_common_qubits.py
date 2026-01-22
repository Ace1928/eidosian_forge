import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_pass_operations_over_no_common_qubits():

    class ExampleGate(cirq.testing.SingleQubitGate):
        pass
    q0, q1 = _make_qubits(2)
    op0 = ExampleGate()(q1)
    ps_before = cirq.PauliString({q0: cirq.Z})
    ps_after = cirq.PauliString({q0: cirq.Z})
    _assert_pass_over([op0], ps_before, ps_after)