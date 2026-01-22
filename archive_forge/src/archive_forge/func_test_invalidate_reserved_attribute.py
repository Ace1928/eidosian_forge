import platform
import time
import unittest
import pytest
from monty.functools import (
def test_invalidate_reserved_attribute(self):
    called = []

    class Bar:

        @cached
        def __bar__(self):
            called.append('bar')
            return 1
    b = Bar()
    assert b.__bar__ == 1
    assert len(called) == 1
    cached.invalidate(b, '__bar__')
    assert b.__bar__ == 1
    assert len(called) == 2