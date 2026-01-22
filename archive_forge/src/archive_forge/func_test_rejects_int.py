import inspect
import unittest
from traits.api import (
def test_rejects_int(self):
    a = MyCallable()
    with self.assertRaises(TraitError) as exception_context:
        a.value = 1
    self.assertIn('must be a callable value', str(exception_context.exception))