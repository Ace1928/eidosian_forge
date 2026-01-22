import platform
import time
import unittest
import pytest
from monty.functools import (
def test_result_shadows_descriptor(self):
    called = []

    class Foo:

        @lazy_property
        def foo(self):
            called.append('foo')
            return 1
    f = Foo()
    assert isinstance(Foo.foo, lazy_property)
    assert f.foo is f.foo
    assert f.foo is f.__dict__['foo']
    assert len(called) == 1
    assert f.foo == 1
    assert f.foo == 1
    assert len(called) == 1
    lazy_property.invalidate(f, 'foo')
    assert f.foo == 1
    assert len(called) == 2
    assert f.foo == 1
    assert f.foo == 1
    assert len(called) == 2