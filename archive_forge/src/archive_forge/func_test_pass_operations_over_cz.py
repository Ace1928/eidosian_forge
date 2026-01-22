import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_pass_operations_over_cz():
    q0, q1 = _make_qubits(2)
    op0 = cirq.CZ(q0, q1)
    ps_before = cirq.PauliString({q0: cirq.Z, q1: cirq.Y})
    ps_after = cirq.PauliString({q1: cirq.Y})
    _assert_pass_over([op0], ps_before, ps_after)