import numpy as np
import pytest
import cirq
def test_expectation_with_coefficient():
    q0 = cirq.NamedQubit('q0')
    d = cirq.ProjectorString({q0: 0}, coefficient=0.6 + 0.4j)
    np.testing.assert_allclose(d.expectation_from_state_vector(np.array([[1.0, 0.0]]), qid_map={q0: 0}), 0.6 + 0.4j)
    np.testing.assert_allclose(d.expectation_from_density_matrix(np.array([[1.0, 0.0], [0.0, 0.0]]), {q0: 0}), 0.6 + 0.4j)