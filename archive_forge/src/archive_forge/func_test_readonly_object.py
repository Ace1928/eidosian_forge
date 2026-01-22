import platform
import time
import unittest
import pytest
from monty.functools import (
def test_readonly_object(self):
    called = []

    class Foo:
        __slots__ = ()

        @lazy_property
        def foo(self):
            called.append('foo')
            return 1
    f = Foo()
    assert len(called) == 0
    with pytest.raises(AttributeError, match="'Foo' object has no attribute '__dict__'"):
        f.foo
    assert len(called) == 0