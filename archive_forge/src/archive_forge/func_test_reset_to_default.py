import unittest
from traits.trait_types import Int
from traits.has_traits import HasTraits
def test_reset_to_default(self):
    foo = Foo(bar=3)
    foo.reset_traits(traits=['bar'])
    self.assertEqual(foo.bar, 4)