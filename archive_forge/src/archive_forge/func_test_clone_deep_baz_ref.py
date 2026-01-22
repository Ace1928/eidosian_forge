import unittest
from traits.api import CList, HasTraits, Instance, Int, List, Str, TraitError
def test_clone_deep_baz_ref(self):
    baz = BazRef()
    for name in ['a', 'b', 'c', 'd']:
        baz.bars.append(Bar(name=name))
    deep_baz = DeepBazBazRef(baz=baz)
    deep_baz_copy = deep_baz.clone_traits()
    self.assertIsNot(deep_baz_copy, deep_baz)
    self.assertIsNot(deep_baz_copy.baz, deep_baz.baz)
    baz_copy = deep_baz_copy.baz
    self.assertIsNot(baz_copy, baz)
    self.assertIsNot(baz_copy.bars, baz.bars)
    self.assertEqual(len(baz_copy.bars), len(baz.bars))
    for bar in baz.bars:
        self.assertIn(bar, baz_copy.bars)