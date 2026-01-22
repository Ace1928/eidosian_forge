import unittest
from traits.api import HasTraits, Instance, Str, Any, Property
def test_Any_circular_references_deep(self):
    bar = BarAny()
    baz = BazAny()
    bar.other = baz
    baz.other = bar
    bar_copy = bar.clone_traits(copy='deep')
    self.assertIsNot(bar_copy, bar)
    self.assertIsNot(bar_copy.other, baz)
    self.assertIsNot(bar_copy.other.other, bar)
    self.assertIs(bar_copy.other.other, bar_copy)