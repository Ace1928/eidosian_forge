import unittest
from traits.api import (
def test_union_incompatible_trait(self):
    with self.assertRaises(ValueError) as exception_context:
        Union(Str(), 'none')
    self.assertEqual(str(exception_context.exception), "Union trait declaration expects a trait type or an instance of trait type or None, but got 'none' instead")