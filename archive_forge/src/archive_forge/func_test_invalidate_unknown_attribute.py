import platform
import time
import unittest
import pytest
from monty.functools import (
def test_invalidate_unknown_attribute(self):
    called = []

    class Bar:

        @cached
        def bar(self):
            called.append('bar')
            return 1
    b = Bar()
    with pytest.raises(AttributeError, match="type object 'Bar' has no attribute 'baz'"):
        lazy_property.invalidate(b, 'baz')