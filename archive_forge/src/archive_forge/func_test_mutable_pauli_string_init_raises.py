import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_mutable_pauli_string_init_raises():
    q = cirq.LineQubit.range(3)
    with pytest.raises(ValueError, match='must be between 1 and 3'):
        _ = cirq.MutablePauliString(pauli_int_dict={q[0]: 0, q[1]: 1, q[2]: 2})