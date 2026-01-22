import inspect
import unittest
from traits.api import (
def test_rejects_non_callable(self):
    with self.assertRaises(TraitError):
        MyBaseCallable(value=Dummy())
    with self.assertRaises(TraitError):
        MyBaseCallable(value=1)