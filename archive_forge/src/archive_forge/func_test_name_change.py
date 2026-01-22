import unittest
from traits.api import HasTraits, Str, Undefined, ReadOnly, Float
def test_name_change(self):
    b = Bar()
    b.name = 'first'
    self.assertEqual(b.name, 'first')