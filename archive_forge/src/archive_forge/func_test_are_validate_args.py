import os
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose
import pytest
from pytest import raises as assert_raises
from scipy.linalg import solve_sylvester
from scipy.linalg import solve_continuous_lyapunov, solve_discrete_lyapunov
from scipy.linalg import solve_continuous_are, solve_discrete_are
from scipy.linalg import block_diag, solve, LinAlgError
from scipy.sparse._sputils import matrix
def test_are_validate_args():

    def test_square_shape():
        nsq = np.ones((3, 2))
        sq = np.eye(3)
        for x in (solve_continuous_are, solve_discrete_are):
            assert_raises(ValueError, x, nsq, 1, 1, 1)
            assert_raises(ValueError, x, sq, sq, nsq, 1)
            assert_raises(ValueError, x, sq, sq, sq, nsq)
            assert_raises(ValueError, x, sq, sq, sq, sq, nsq)

    def test_compatible_sizes():
        nsq = np.ones((3, 2))
        sq = np.eye(4)
        for x in (solve_continuous_are, solve_discrete_are):
            assert_raises(ValueError, x, sq, nsq, 1, 1)
            assert_raises(ValueError, x, sq, sq, sq, sq, sq, nsq)
            assert_raises(ValueError, x, sq, sq, np.eye(3), sq)
            assert_raises(ValueError, x, sq, sq, sq, np.eye(3))
            assert_raises(ValueError, x, sq, sq, sq, sq, np.eye(3))

    def test_symmetry():
        nsym = np.arange(9).reshape(3, 3)
        sym = np.eye(3)
        for x in (solve_continuous_are, solve_discrete_are):
            assert_raises(ValueError, x, sym, sym, nsym, sym)
            assert_raises(ValueError, x, sym, sym, sym, nsym)

    def test_singularity():
        sing = np.full((3, 3), 1000000000000.0)
        sing[2, 2] -= 1
        sq = np.eye(3)
        for x in (solve_continuous_are, solve_discrete_are):
            assert_raises(ValueError, x, sq, sq, sq, sq, sing)
        assert_raises(ValueError, solve_continuous_are, sq, sq, sq, sing)

    def test_finiteness():
        nm = np.full((2, 2), np.nan)
        sq = np.eye(2)
        for x in (solve_continuous_are, solve_discrete_are):
            assert_raises(ValueError, x, nm, sq, sq, sq)
            assert_raises(ValueError, x, sq, nm, sq, sq)
            assert_raises(ValueError, x, sq, sq, nm, sq)
            assert_raises(ValueError, x, sq, sq, sq, nm)
            assert_raises(ValueError, x, sq, sq, sq, sq, nm)
            assert_raises(ValueError, x, sq, sq, sq, sq, sq, nm)