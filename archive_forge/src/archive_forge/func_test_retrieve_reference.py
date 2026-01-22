import unittest
from traits.api import CList, HasTraits, Instance, Int, List, Str, TraitError
def test_retrieve_reference(self):
    f = Foo(l=['initial', 'value'])
    l = f.l
    self.assertIs(l, f.l)
    l.append('change')
    self.assertEqual(f.l, ['initial', 'value', 'change'])
    f.l.append('more change')
    self.assertEqual(l, ['initial', 'value', 'change', 'more change'])