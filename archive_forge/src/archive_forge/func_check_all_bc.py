import warnings
import io
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
def check_all_bc(self, x, y, axis):
    deriv_shape = list(y.shape)
    del deriv_shape[axis]
    first_deriv = np.empty(deriv_shape)
    first_deriv.fill(2)
    second_deriv = np.empty(deriv_shape)
    second_deriv.fill(-1)
    bc_all = ['not-a-knot', 'natural', 'clamped', (1, first_deriv), (2, second_deriv)]
    for bc in bc_all[:3]:
        S = CubicSpline(x, y, axis=axis, bc_type=bc)
        self.check_correctness(S, bc, bc)
    for bc_start in bc_all:
        for bc_end in bc_all:
            S = CubicSpline(x, y, axis=axis, bc_type=(bc_start, bc_end))
            self.check_correctness(S, bc_start, bc_end, tol=2e-14)