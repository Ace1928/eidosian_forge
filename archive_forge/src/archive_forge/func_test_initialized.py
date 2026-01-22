import unittest
from traits.api import CList, HasTraits, Instance, Int, List, Str, TraitError
def test_initialized(self):
    f = Foo()
    self.assertNotEqual(f.l, None)
    self.assertEqual(len(f.l), 0)