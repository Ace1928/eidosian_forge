import itertools
from typing import Optional
import pytest
import numpy as np
import cirq
import cirq.testing
from cirq import linalg
def test_step_result_density_matrix():
    q0, q1 = cirq.LineQubit.range(2)
    step_result = BasicStateVector({q0: 0, q1: 1})
    rho = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    np.testing.assert_array_almost_equal(rho, step_result.density_matrix_of([q0, q1]))
    np.testing.assert_array_almost_equal(rho, step_result.density_matrix_of())
    rho_ind_rev = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
    np.testing.assert_array_almost_equal(rho_ind_rev, step_result.density_matrix_of([q1, q0]))
    single_rho = np.array([[0, 0], [0, 1]])
    np.testing.assert_array_almost_equal(single_rho, step_result.density_matrix_of([q1]))