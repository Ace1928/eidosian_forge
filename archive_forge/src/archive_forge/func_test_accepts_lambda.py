import inspect
import unittest
from traits.api import (
def test_accepts_lambda(self):
    func = lambda v: v + 1
    a = MyCallable(value=func)
    self.assertIs(a.value, func)