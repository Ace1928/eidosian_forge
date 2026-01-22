import platform
import time
import unittest
import pytest
from monty.functools import (
def test_invalidate_nonlazy_attribute(self):
    called = []

    class Foo:

        def foo(self):
            called.append('foo')
            return 1
    f = Foo()
    with pytest.raises(AttributeError, match="'Foo.foo' is not a lazy_property attribute"):
        lazy_property.invalidate(f, 'foo')