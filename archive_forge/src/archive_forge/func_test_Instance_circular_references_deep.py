import unittest
from traits.api import HasTraits, Instance, Str, Any, Property
def test_Instance_circular_references_deep(self):
    ref = Foo(s='ref')
    bar_unique = Foo(s='bar.foo')
    shared = Foo(s='shared')
    baz_unique = Foo(s='baz.unique')
    baz = BazInstance()
    baz.unique = baz_unique
    baz.shared = shared
    baz.ref = ref
    bar = BarInstance()
    bar.unique = bar_unique
    bar.shared = shared
    bar.ref = ref
    bar.other = baz
    baz.other = bar
    baz_copy = baz.clone_traits(copy='deep')
    self.assertIsNot(baz_copy, baz)
    self.assertIsNot(baz_copy.other, bar)
    self.assertIsNot(baz_copy.unique, baz.unique)
    self.assertIsNot(baz_copy.shared, baz.shared)
    bar_copy = baz_copy.other
    self.assertIsNot(bar_copy.unique, bar.unique)
    self.assertIs(baz_copy.ref, bar_copy.ref)
    self.assertIs(bar_copy.ref, ref)
    self.assertIsNot(bar_copy.other, baz_copy)
    self.assertIs(bar_copy.other, baz)
    self.assertIsNot(bar_copy.shared, baz.shared)
    self.assertIs(bar_copy.shared, baz_copy.shared)