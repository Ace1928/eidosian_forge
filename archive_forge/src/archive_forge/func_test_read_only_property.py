import unittest
from traits.api import HasTraits
def test_read_only_property(self):
    model = Model()
    self.assertEqual(model.read_only, 1729)
    with self.assertRaises(AttributeError):
        model.read_only = 2034
    with self.assertRaises(AttributeError):
        del model.read_only