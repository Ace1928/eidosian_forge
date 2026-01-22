import unittest
from traits.api import CList, HasTraits, Instance, Int, List, Str, TraitError
def test_clone_ref(self):
    baz = BazRef()
    for name in ['a', 'b', 'c', 'd']:
        baz.bars.append(Bar(name=name))
    baz_copy = baz.clone_traits()
    self.assertIsNot(baz_copy, baz)
    self.assertIsNot(baz_copy.bars, baz.bars)
    self.assertEqual(len(baz_copy.bars), len(baz.bars))
    for bar in baz.bars:
        self.assertIn(bar, baz_copy.bars)