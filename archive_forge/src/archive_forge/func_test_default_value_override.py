import unittest
from traits.trait_types import Int
from traits.has_traits import HasTraits
def test_default_value_override(self):
    foo = Foo(bar=3)
    self.assertEqual(foo.bar, 3)