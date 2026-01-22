import collections
from typing import Union
import numpy as np
import pytest
import sympy
import sympy.parsing.sympy_parser as sympy_parser
import cirq
import cirq.testing
def test_expectation_from_state_vector_invalid_input():
    q0, q1, q2, q3 = cirq.LineQubit.range(4)
    psum = cirq.X(q0) + 2 * cirq.Y(q1) + 3 * cirq.Z(q3)
    q_map = {q0: 0, q1: 1, q3: 2}
    wf = np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.complex64)
    im_psum = (1j + 1) * psum
    with pytest.raises(NotImplementedError, match='non-Hermitian'):
        im_psum.expectation_from_state_vector(wf, q_map)
    with pytest.raises(TypeError, match='dtype'):
        psum.expectation_from_state_vector(np.array([1, 0], dtype=int), q_map)
    with pytest.raises(TypeError, match='mapping'):
        psum.expectation_from_state_vector(wf, 'bad type')
    with pytest.raises(TypeError, match='mapping'):
        psum.expectation_from_state_vector(wf, {'bad key': 1})
    with pytest.raises(TypeError, match='mapping'):
        psum.expectation_from_state_vector(wf, {q0: 'bad value'})
    with pytest.raises(ValueError, match='complete'):
        psum.expectation_from_state_vector(wf, {q0: 0})
    with pytest.raises(ValueError, match='complete'):
        psum.expectation_from_state_vector(wf, {q0: 0, q2: 2})
    with pytest.raises(ValueError, match='indices'):
        psum.expectation_from_state_vector(wf, {q0: -1, q1: 1, q3: 2})
    with pytest.raises(ValueError, match='indices'):
        psum.expectation_from_state_vector(wf, {q0: 0, q1: 3, q3: 2})
    with pytest.raises(ValueError, match='indices'):
        psum.expectation_from_state_vector(wf, {q0: 0, q1: 0, q3: 2})
    with pytest.raises(ValueError, match='9'):
        psum.expectation_from_state_vector(np.arange(9, dtype=np.complex64), q_map)
    q_map_2 = {q0: 0, q1: 1, q2: 2, q3: 3}
    with pytest.raises(ValueError, match='normalized'):
        psum.expectation_from_state_vector(np.arange(16, dtype=np.complex64), q_map_2)
    wf = np.arange(16, dtype=np.complex64) / np.linalg.norm(np.arange(16))
    with pytest.raises(ValueError, match='shape'):
        psum.expectation_from_state_vector(wf.reshape((16, 1)), q_map_2)
    with pytest.raises(ValueError, match='shape'):
        psum.expectation_from_state_vector(wf.reshape((4, 4, 1)), q_map_2)