import inspect
import unittest
from traits.api import (
def test_none_not_allowed_in_compound(self):
    obj = MyCallable()
    with self.assertRaises(TraitError):
        obj.non_none_callable_or_str = None