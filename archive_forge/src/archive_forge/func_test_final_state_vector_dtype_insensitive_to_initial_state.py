import collections
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_final_state_vector_dtype_insensitive_to_initial_state():
    assert cirq.final_state_vector(cirq.X).dtype == np.complex64
    assert cirq.final_state_vector(cirq.X, initial_state=0).dtype == np.complex64
    assert cirq.final_state_vector(cirq.X, initial_state=[np.sqrt(0.5), np.sqrt(0.5)]).dtype == np.complex64
    assert cirq.final_state_vector(cirq.X, initial_state=np.array([np.sqrt(0.5), np.sqrt(0.5)])).dtype == np.complex64
    for t in [np.int32, np.float32, np.float64, np.complex64]:
        assert cirq.final_state_vector(cirq.X, initial_state=np.array([1, 0], dtype=t)).dtype == np.complex64
        assert cirq.final_state_vector(cirq.X, initial_state=np.array([1, 0], dtype=t), dtype=np.complex128).dtype == np.complex128