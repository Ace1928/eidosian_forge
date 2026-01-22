import inspect
import unittest
from traits.api import (
def test_old_style_callable(self):

    class MyCallable(HasTraits):
        value = OldCallable()
    obj = MyCallable()
    obj.value = None
    self.assertIsNone(obj.value)