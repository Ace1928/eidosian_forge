import pytest
import itertools
from scipy.stats import (betabinom, betanbinom, hypergeom, nhypergeom,
import numpy as np
from numpy.testing import (
from scipy.special import binom as special_binom
from scipy.optimize import root_scalar
from scipy.integrate import quad
def test_nchypergeom_wallenius_naive(self):
    np.random.seed(2)
    shape = (2, 4, 3)
    max_m = 100
    m1 = np.random.randint(1, max_m, size=shape)
    m2 = np.random.randint(1, max_m, size=shape)
    N = m1 + m2
    n = randint.rvs(0, N, size=N.shape)
    xl = np.maximum(0, n - m2)
    xu = np.minimum(n, m1)
    x = randint.rvs(xl, xu, size=xl.shape)
    w = np.random.rand(*x.shape) * 2

    def support(N, m1, n, w):
        m2 = N - m1
        xl = np.maximum(0, n - m2)
        xu = np.minimum(n, m1)
        return (xl, xu)

    @np.vectorize
    def mean(N, m1, n, w):
        m2 = N - m1
        xl, xu = support(N, m1, n, w)

        def fun(u):
            return u / m1 + (1 - (n - u) / m2) ** w - 1
        return root_scalar(fun, bracket=(xl, xu)).root
    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning, message='invalid value encountered in mean')
        assert_allclose(nchypergeom_wallenius.mean(N, m1, n, w), mean(N, m1, n, w), rtol=0.02)

    @np.vectorize
    def variance(N, m1, n, w):
        m2 = N - m1
        u = mean(N, m1, n, w)
        a = u * (m1 - u)
        b = (n - u) * (u + m2 - n)
        return N * a * b / ((N - 1) * (m1 * b + m2 * a))
    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning, message='invalid value encountered in mean')
        assert_allclose(nchypergeom_wallenius.stats(N, m1, n, w, moments='v'), variance(N, m1, n, w), rtol=0.05)

    @np.vectorize
    def pmf(x, N, m1, n, w):
        m2 = N - m1
        xl, xu = support(N, m1, n, w)

        def integrand(t):
            D = w * (m1 - x) + (m2 - (n - x))
            res = (1 - t ** (w / D)) ** x * (1 - t ** (1 / D)) ** (n - x)
            return res

        def f(x):
            t1 = special_binom(m1, x)
            t2 = special_binom(m2, n - x)
            the_integral = quad(integrand, 0, 1, epsrel=1e-16, epsabs=1e-16)
            return t1 * t2 * the_integral[0]
        return f(x)
    pmf0 = pmf(x, N, m1, n, w)
    pmf1 = nchypergeom_wallenius.pmf(x, N, m1, n, w)
    atol, rtol = (1e-06, 1e-06)
    i = np.abs(pmf1 - pmf0) < atol + rtol * np.abs(pmf0)
    assert i.sum() > np.prod(shape) / 2
    for N, m1, n, w in zip(N[~i], m1[~i], n[~i], w[~i]):
        m2 = N - m1
        xl, xu = support(N, m1, n, w)
        x = np.arange(xl, xu + 1)
        assert pmf(x, N, m1, n, w).sum() < 0.5
        assert_allclose(nchypergeom_wallenius.pmf(x, N, m1, n, w).sum(), 1)