import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
def test_expectation_from_state_vector_check_preconditions():
    q0, q1, q2, q3 = cirq.LineQubit.range(4)
    psum = cirq.X(q0) + 2 * cirq.Y(q1) + 3 * cirq.Z(q3)
    q_map = {q0: 0, q1: 1, q2: 2, q3: 3}
    with pytest.raises(ValueError, match='normalized'):
        psum.expectation_from_state_vector(np.arange(16, dtype=np.complex64), q_map)
    _ = psum.expectation_from_state_vector(np.arange(16, dtype=np.complex64), q_map, check_preconditions=False)