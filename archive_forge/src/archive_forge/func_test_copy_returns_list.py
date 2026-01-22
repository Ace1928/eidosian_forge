import unittest
from traits.api import CList, HasTraits, Instance, Int, List, Str, TraitError
def test_copy_returns_list(self):
    f = Foo()
    f.l = ['a', 'c', 'b', 'd']
    l_copy = f.l.copy()
    self.assertEqual(type(l_copy), list)