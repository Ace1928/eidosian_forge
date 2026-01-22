import unittest
from traits.api import (
def test_trait_names_returned_by_visible_traits(self):
    b = Bar()
    self.assertEqual(sorted(b.visible_traits()), sorted(['PubT1', 'PrivT2']))