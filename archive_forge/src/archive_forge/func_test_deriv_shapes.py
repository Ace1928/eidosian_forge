import warnings
import io
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
def test_deriv_shapes():

    def krogh_deriv(x, y, axis=0):
        return KroghInterpolator(x, y, axis).derivative

    def bary_deriv(x, y, axis=0):
        return BarycentricInterpolator(x, y, axis).derivative

    def pchip_deriv(x, y, axis=0):
        return pchip(x, y, axis).derivative()

    def pchip_deriv2(x, y, axis=0):
        return pchip(x, y, axis).derivative(2)

    def pchip_antideriv(x, y, axis=0):
        return pchip(x, y, axis).antiderivative()

    def pchip_antideriv2(x, y, axis=0):
        return pchip(x, y, axis).antiderivative(2)

    def pchip_deriv_inplace(x, y, axis=0):

        class P(PchipInterpolator):

            def __call__(self, x):
                return PchipInterpolator.__call__(self, x, 1)
            pass
        return P(x, y, axis)

    def akima_deriv(x, y, axis=0):
        return Akima1DInterpolator(x, y, axis).derivative()

    def akima_antideriv(x, y, axis=0):
        return Akima1DInterpolator(x, y, axis).antiderivative()

    def cspline_deriv(x, y, axis=0):
        return CubicSpline(x, y, axis).derivative()

    def cspline_antideriv(x, y, axis=0):
        return CubicSpline(x, y, axis).antiderivative()

    def bspl_deriv(x, y, axis=0):
        return make_interp_spline(x, y, axis=axis).derivative()

    def bspl_antideriv(x, y, axis=0):
        return make_interp_spline(x, y, axis=axis).antiderivative()
    for ip in [krogh_deriv, bary_deriv, pchip_deriv, pchip_deriv2, pchip_deriv_inplace, pchip_antideriv, pchip_antideriv2, akima_deriv, akima_antideriv, cspline_deriv, cspline_antideriv, bspl_deriv, bspl_antideriv]:
        for s1 in SHAPES:
            for s2 in SHAPES:
                for axis in range(-len(s2), len(s2)):
                    check_shape(ip, s1, s2, (), axis)