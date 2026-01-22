import unittest
from traits.api import (
from traits.testing.optional_dependencies import numpy, requires_numpy
def test_accepts_int_like(self):
    self.model.percentage = IntLike(35)
    self.assertIs(type(self.model.percentage), int)
    self.assertEqual(self.model.percentage, 35)
    with self.assertRaises(TraitError):
        self.model.percentage = IntLike(-1)
    with self.assertRaises(TraitError):
        self.model.percentage = IntLike(101)