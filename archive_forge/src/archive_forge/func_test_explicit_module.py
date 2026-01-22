import pytest
import pickle
from numpy.testing import assert_equal
from scipy._lib._bunch import _make_tuple_bunch
def test_explicit_module(self):
    m = 'some.module.name'
    Foo = _make_tuple_bunch('Foo', ['x'], ['a', 'b'], module=m)
    foo = Foo(x=1, a=355, b=113)
    assert_equal(Foo.__module__, m)
    assert_equal(foo.__module__, m)