import itertools
import numpy as np
import sys
from collections import namedtuple
from io import StringIO
from numba import njit, typeof, prange
from numba.core import (
from numba.tests.support import (TestCase, tag, skip_parfors_unsupported,
from numba.parfors.array_analysis import EquivSet, ArrayAnalysis
from numba.core.compiler import Compiler, Flags, PassManager
from numba.core.ir_utils import remove_dead
from numba.core.untyped_passes import (ExtractByteCode, TranslateByteCode, FixupArgs,
from numba.core.typed_passes import (NopythonTypeInference, AnnotateTypes,
from numba.core.compiler_machinery import FunctionPass, PassManager, register_pass
from numba.experimental import jitclass
import unittest
@skip_unless_scipy
def test_numpy_calls(self):

    def test_zeros(n):
        a = np.zeros(n)
        b = np.zeros((n, n))
        c = np.zeros(shape=(n, n))
    self._compile_and_test(test_zeros, (types.intp,), equivs=[self.with_equiv('a', 'n'), self.with_equiv('b', ('n', 'n')), self.with_equiv('b', 'c')])

    def test_0d_array(n):
        a = np.array(1)
        b = np.ones(2)
        return a + b
    self._compile_and_test(test_0d_array, (types.intp,), equivs=[self.without_equiv('a', 'b')], asserts=[self.without_shapecall('a')])

    def test_ones(n):
        a = np.ones(n)
        b = np.ones((n, n))
        c = np.ones(shape=(n, n))
    self._compile_and_test(test_ones, (types.intp,), equivs=[self.with_equiv('a', 'n'), self.with_equiv('b', ('n', 'n')), self.with_equiv('b', 'c')])

    def test_empty(n):
        a = np.empty(n)
        b = np.empty((n, n))
        c = np.empty(shape=(n, n))
    self._compile_and_test(test_empty, (types.intp,), equivs=[self.with_equiv('a', 'n'), self.with_equiv('b', ('n', 'n')), self.with_equiv('b', 'c')])

    def test_eye(n):
        a = np.eye(n)
        b = np.eye(N=n)
        c = np.eye(N=n, M=n)
        d = np.eye(N=n, M=n + 1)
    self._compile_and_test(test_eye, (types.intp,), equivs=[self.with_equiv('a', ('n', 'n')), self.with_equiv('b', ('n', 'n')), self.with_equiv('b', 'c'), self.without_equiv('b', 'd')])

    def test_identity(n):
        a = np.identity(n)
    self._compile_and_test(test_identity, (types.intp,), equivs=[self.with_equiv('a', ('n', 'n'))])

    def test_diag(n):
        a = np.identity(n)
        b = np.diag(a)
        c = np.diag(b)
        d = np.diag(a, k=1)
    self._compile_and_test(test_diag, (types.intp,), equivs=[self.with_equiv('b', ('n',)), self.with_equiv('c', ('n', 'n'))], asserts=[self.with_shapecall('d'), self.without_shapecall('c')])

    def test_array_like(a):
        b = np.empty_like(a)
        c = np.zeros_like(a)
        d = np.ones_like(a)
        e = np.full_like(a, 1)
        f = np.asfortranarray(a)
    self._compile_and_test(test_array_like, (types.Array(types.intp, 2, 'C'),), equivs=[self.with_equiv('a', 'b', 'd', 'e', 'f')], asserts=[self.with_shapecall('a'), self.without_shapecall('b')])

    def test_reshape(n):
        a = np.ones(n * n)
        b = a.reshape((n, n))
        return a.sum() + b.sum()
    self._compile_and_test(test_reshape, (types.intp,), equivs=[self.with_equiv('b', ('n', 'n'))], asserts=[self.without_shapecall('b')])

    def test_transpose(m, n):
        a = np.ones((m, n))
        b = a.T
        c = a.transpose()
    self._compile_and_test(test_transpose, (types.intp, types.intp), equivs=[self.with_equiv('a', ('m', 'n')), self.with_equiv('b', ('n', 'm')), self.with_equiv('c', ('n', 'm'))])

    def test_transpose_3d(m, n, k):
        a = np.ones((m, n, k))
        b = a.T
        c = a.transpose()
        d = a.transpose(2, 0, 1)
        dt = a.transpose((2, 0, 1))
        e = a.transpose(0, 2, 1)
        et = a.transpose((0, 2, 1))
    self._compile_and_test(test_transpose_3d, (types.intp, types.intp, types.intp), equivs=[self.with_equiv('a', ('m', 'n', 'k')), self.with_equiv('b', ('k', 'n', 'm')), self.with_equiv('c', ('k', 'n', 'm')), self.with_equiv('d', ('k', 'm', 'n')), self.with_equiv('dt', ('k', 'm', 'n')), self.with_equiv('e', ('m', 'k', 'n')), self.with_equiv('et', ('m', 'k', 'n'))])

    def test_real_imag_attr(m, n):
        a = np.ones((m, n))
        b = a.real
        c = a.imag
    self._compile_and_test(test_real_imag_attr, (types.intp, types.intp), equivs=[self.with_equiv('a', ('m', 'n')), self.with_equiv('b', ('m', 'n')), self.with_equiv('c', ('m', 'n'))])

    def test_random(n):
        a0 = np.random.rand(n)
        a1 = np.random.rand(n, n)
        b0 = np.random.randn(n)
        b1 = np.random.randn(n, n)
        c0 = np.random.ranf(n)
        c1 = np.random.ranf((n, n))
        c2 = np.random.ranf(size=(n, n))
        d0 = np.random.random_sample(n)
        d1 = np.random.random_sample((n, n))
        d2 = np.random.random_sample(size=(n, n))
        e0 = np.random.sample(n)
        e1 = np.random.sample((n, n))
        e2 = np.random.sample(size=(n, n))
        f0 = np.random.random(n)
        f1 = np.random.random((n, n))
        f2 = np.random.random(size=(n, n))
        g0 = np.random.standard_normal(n)
        g1 = np.random.standard_normal((n, n))
        g2 = np.random.standard_normal(size=(n, n))
        h0 = np.random.chisquare(10, n)
        h1 = np.random.chisquare(10, (n, n))
        h2 = np.random.chisquare(10, size=(n, n))
        i0 = np.random.weibull(10, n)
        i1 = np.random.weibull(10, (n, n))
        i2 = np.random.weibull(10, size=(n, n))
        j0 = np.random.power(10, n)
        j1 = np.random.power(10, (n, n))
        j2 = np.random.power(10, size=(n, n))
        k0 = np.random.geometric(0.1, n)
        k1 = np.random.geometric(0.1, (n, n))
        k2 = np.random.geometric(0.1, size=(n, n))
        l0 = np.random.exponential(10, n)
        l1 = np.random.exponential(10, (n, n))
        l2 = np.random.exponential(10, size=(n, n))
        m0 = np.random.poisson(10, n)
        m1 = np.random.poisson(10, (n, n))
        m2 = np.random.poisson(10, size=(n, n))
        n0 = np.random.rayleigh(10, n)
        n1 = np.random.rayleigh(10, (n, n))
        n2 = np.random.rayleigh(10, size=(n, n))
        o0 = np.random.normal(0, 1, n)
        o1 = np.random.normal(0, 1, (n, n))
        o2 = np.random.normal(0, 1, size=(n, n))
        p0 = np.random.uniform(0, 1, n)
        p1 = np.random.uniform(0, 1, (n, n))
        p2 = np.random.uniform(0, 1, size=(n, n))
        q0 = np.random.beta(0.1, 1, n)
        q1 = np.random.beta(0.1, 1, (n, n))
        q2 = np.random.beta(0.1, 1, size=(n, n))
        r0 = np.random.binomial(0, 1, n)
        r1 = np.random.binomial(0, 1, (n, n))
        r2 = np.random.binomial(0, 1, size=(n, n))
        s0 = np.random.f(0.1, 1, n)
        s1 = np.random.f(0.1, 1, (n, n))
        s2 = np.random.f(0.1, 1, size=(n, n))
        t0 = np.random.gamma(0.1, 1, n)
        t1 = np.random.gamma(0.1, 1, (n, n))
        t2 = np.random.gamma(0.1, 1, size=(n, n))
        u0 = np.random.lognormal(0, 1, n)
        u1 = np.random.lognormal(0, 1, (n, n))
        u2 = np.random.lognormal(0, 1, size=(n, n))
        v0 = np.random.laplace(0, 1, n)
        v1 = np.random.laplace(0, 1, (n, n))
        v2 = np.random.laplace(0, 1, size=(n, n))
        w0 = np.random.randint(0, 10, n)
        w1 = np.random.randint(0, 10, (n, n))
        w2 = np.random.randint(0, 10, size=(n, n))
        x0 = np.random.triangular(-3, 0, 10, n)
        x1 = np.random.triangular(-3, 0, 10, (n, n))
        x2 = np.random.triangular(-3, 0, 10, size=(n, n))
    last = ord('x') + 1
    vars1d = [('n',)] + [chr(x) + '0' for x in range(ord('a'), last)]
    vars2d = [('n', 'n')] + [chr(x) + '1' for x in range(ord('a'), last)]
    vars2d += [chr(x) + '1' for x in range(ord('c'), last)]
    self._compile_and_test(test_random, (types.intp,), equivs=[self.with_equiv(*vars1d), self.with_equiv(*vars2d)])

    def test_concatenate(m, n):
        a = np.ones(m)
        b = np.ones(n)
        c = np.concatenate((a, b))
        d = np.ones((2, n))
        e = np.ones((3, n))
        f = np.concatenate((d, e))
        i = np.ones((m, 2))
        j = np.ones((m, 3))
        k = np.concatenate((i, j), axis=1)
        l = np.ones((m, n))
        o = np.ones((m, n))
        p = np.concatenate((l, o))
    self._compile_and_test(test_concatenate, (types.intp, types.intp), equivs=[self.with_equiv('f', (5, 'n')), self.with_equiv('k', ('m', 5))], asserts=[self.with_shapecall('c'), self.without_shapecall('f'), self.without_shapecall('k'), self.with_shapecall('p')])

    def test_vsd_stack():
        k = np.ones((2,))
        l = np.ones((2, 3))
        o = np.ones((2, 3, 4))
        p = np.vstack((k, k))
        q = np.vstack((l, l))
        r = np.hstack((k, k))
        s = np.hstack((l, l))
        t = np.dstack((k, k))
        u = np.dstack((l, l))
        v = np.dstack((o, o))
    self._compile_and_test(test_vsd_stack, (), equivs=[self.with_equiv('p', (2, 2)), self.with_equiv('q', (4, 3)), self.with_equiv('r', (4,)), self.with_equiv('s', (2, 6)), self.with_equiv('t', (1, 2, 2)), self.with_equiv('u', (2, 3, 2)), self.with_equiv('v', (2, 3, 8))])

    def test_stack(m, n):
        a = np.ones(m)
        b = np.ones(n)
        c = np.stack((a, b))
        d = np.ones((m, n))
        e = np.ones((m, n))
        f = np.stack((d, e))
        g = np.stack((d, e), axis=0)
        h = np.stack((d, e), axis=1)
        i = np.stack((d, e), axis=2)
        j = np.stack((d, e), axis=-1)
    self._compile_and_test(test_stack, (types.intp, types.intp), equivs=[self.with_equiv('m', 'n'), self.with_equiv('c', (2, 'm')), self.with_equiv('f', 'g', (2, 'm', 'n')), self.with_equiv('h', ('m', 2, 'n')), self.with_equiv('i', 'j', ('m', 'n', 2))])

    def test_linspace(m, n):
        a = np.linspace(m, n)
        b = np.linspace(m, n, 10)
    self._compile_and_test(test_linspace, (types.float64, types.float64), equivs=[self.with_equiv('a', (50,)), self.with_equiv('b', (10,))])

    def test_dot(l, m, n):
        a = np.dot(np.ones(1), np.ones(1))
        b = np.dot(np.ones(2), np.ones((2, 3)))
        e = np.dot(np.ones((1, 2)), np.ones(2))
        h = np.dot(np.ones((2, 3)), np.ones((3, 4)))
        i = np.dot(np.ones((m, n)), np.ones((n, m)))
        j = np.dot(np.ones((m, m)), np.ones((l, l)))
    self._compile_and_test(test_dot, (types.intp, types.intp, types.intp), equivs=[self.without_equiv('a', (1,)), self.with_equiv('b', (3,)), self.with_equiv('e', (1,)), self.with_equiv('h', (2, 4)), self.with_equiv('i', ('m', 'm')), self.with_equiv('j', ('m', 'm'))], asserts=[self.with_assert('m', 'l')])

    def test_broadcast(m, n):
        a = np.ones((m, n))
        b = np.ones(n)
        c = a + b
        d = np.ones((1, n))
        e = a + c - d
    self._compile_and_test(test_broadcast, (types.intp, types.intp), equivs=[self.with_equiv('a', 'c', 'e')], asserts=None)

    def test_global_tuple():
        a = np.ones(GVAL2)
        b = np.ones(GVAL2)
    self._compile_and_test(test_global_tuple, (), equivs=[self.with_equiv('a', 'b')], asserts=None)