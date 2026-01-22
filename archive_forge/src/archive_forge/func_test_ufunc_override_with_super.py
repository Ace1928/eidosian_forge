import platform
import warnings
import fnmatch
import itertools
import pytest
import sys
import os
import operator
from fractions import Fraction
from functools import reduce
from collections import namedtuple
import numpy.core.umath as ncu
from numpy.core import _umath_tests as ncu_tests
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _glibc_older_than
def test_ufunc_override_with_super(self):

    class A(np.ndarray):

        def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
            args = []
            in_no = []
            for i, input_ in enumerate(inputs):
                if isinstance(input_, A):
                    in_no.append(i)
                    args.append(input_.view(np.ndarray))
                else:
                    args.append(input_)
            outputs = out
            out_no = []
            if outputs:
                out_args = []
                for j, output in enumerate(outputs):
                    if isinstance(output, A):
                        out_no.append(j)
                        out_args.append(output.view(np.ndarray))
                    else:
                        out_args.append(output)
                kwargs['out'] = tuple(out_args)
            else:
                outputs = (None,) * ufunc.nout
            info = {}
            if in_no:
                info['inputs'] = in_no
            if out_no:
                info['outputs'] = out_no
            results = super().__array_ufunc__(ufunc, method, *args, **kwargs)
            if results is NotImplemented:
                return NotImplemented
            if method == 'at':
                if isinstance(inputs[0], A):
                    inputs[0].info = info
                return
            if ufunc.nout == 1:
                results = (results,)
            results = tuple((np.asarray(result).view(A) if output is None else output for result, output in zip(results, outputs)))
            if results and isinstance(results[0], A):
                results[0].info = info
            return results[0] if len(results) == 1 else results

    class B:

        def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
            if any((isinstance(input_, A) for input_ in inputs)):
                return 'A!'
            else:
                return NotImplemented
    d = np.arange(5.0)
    a = np.arange(5.0).view(A)
    b = np.sin(a)
    check = np.sin(d)
    assert_(np.all(check == b))
    assert_equal(b.info, {'inputs': [0]})
    b = np.sin(d, out=(a,))
    assert_(np.all(check == b))
    assert_equal(b.info, {'outputs': [0]})
    assert_(b is a)
    a = np.arange(5.0).view(A)
    b = np.sin(a, out=a)
    assert_(np.all(check == b))
    assert_equal(b.info, {'inputs': [0], 'outputs': [0]})
    a = np.arange(5.0).view(A)
    b1, b2 = np.modf(a)
    assert_equal(b1.info, {'inputs': [0]})
    b1, b2 = np.modf(d, out=(None, a))
    assert_(b2 is a)
    assert_equal(b1.info, {'outputs': [1]})
    a = np.arange(5.0).view(A)
    b = np.arange(5.0).view(A)
    c1, c2 = np.modf(a, out=(a, b))
    assert_(c1 is a)
    assert_(c2 is b)
    assert_equal(c1.info, {'inputs': [0], 'outputs': [0, 1]})
    a = np.arange(5.0).view(A)
    b = np.arange(5.0).view(A)
    c = np.add(a, b, out=a)
    assert_(c is a)
    assert_equal(c.info, {'inputs': [0, 1], 'outputs': [0]})
    a = np.arange(5.0)
    b = B()
    assert_(a.__array_ufunc__(np.add, '__call__', a, b) is NotImplemented)
    assert_(b.__array_ufunc__(np.add, '__call__', a, b) is NotImplemented)
    assert_raises(TypeError, np.add, a, b)
    a = a.view(A)
    assert_(a.__array_ufunc__(np.add, '__call__', a, b) is NotImplemented)
    assert_(b.__array_ufunc__(np.add, '__call__', a, b) == 'A!')
    assert_(np.add(a, b) == 'A!')
    d = np.array([[1, 2, 3], [1, 2, 3]])
    a = d.view(A)
    c = a.any()
    check = d.any()
    assert_equal(c, check)
    assert_(c.info, {'inputs': [0]})
    c = a.max()
    check = d.max()
    assert_equal(c, check)
    assert_(c.info, {'inputs': [0]})
    b = np.array(0).view(A)
    c = a.max(out=b)
    assert_equal(c, check)
    assert_(c is b)
    assert_(c.info, {'inputs': [0], 'outputs': [0]})
    check = a.max(axis=0)
    b = np.zeros_like(check).view(A)
    c = a.max(axis=0, out=b)
    assert_equal(c, check)
    assert_(c is b)
    assert_(c.info, {'inputs': [0], 'outputs': [0]})
    check = np.add.reduce(d, axis=1)
    c = np.add.reduce(a, axis=1)
    assert_equal(c, check)
    assert_(c.info, {'inputs': [0]})
    b = np.zeros_like(c)
    c = np.add.reduce(a, 1, None, b)
    assert_equal(c, check)
    assert_(c is b)
    assert_(c.info, {'inputs': [0], 'outputs': [0]})
    check = np.add.accumulate(d, axis=0)
    c = np.add.accumulate(a, axis=0)
    assert_equal(c, check)
    assert_(c.info, {'inputs': [0]})
    b = np.zeros_like(c)
    c = np.add.accumulate(a, 0, None, b)
    assert_equal(c, check)
    assert_(c is b)
    assert_(c.info, {'inputs': [0], 'outputs': [0]})
    indices = [0, 2, 1]
    check = np.add.reduceat(d, indices, axis=1)
    c = np.add.reduceat(a, indices, axis=1)
    assert_equal(c, check)
    assert_(c.info, {'inputs': [0]})
    b = np.zeros_like(c)
    c = np.add.reduceat(a, indices, 1, None, b)
    assert_equal(c, check)
    assert_(c is b)
    assert_(c.info, {'inputs': [0], 'outputs': [0]})
    d = np.array([[1, 2, 3], [1, 2, 3]])
    check = d.copy()
    a = d.copy().view(A)
    np.add.at(check, ([0, 1], [0, 2]), 1.0)
    np.add.at(a, ([0, 1], [0, 2]), 1.0)
    assert_equal(a, check)
    assert_(a.info, {'inputs': [0]})
    b = np.array(1.0).view(A)
    a = d.copy().view(A)
    np.add.at(a, ([0, 1], [0, 2]), b)
    assert_equal(a, check)
    assert_(a.info, {'inputs': [0, 2]})