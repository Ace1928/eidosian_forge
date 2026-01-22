import unittest
from traits.api import (
def test_type_query(self):
    names = self.foo.copyable_trait_names(**{'type': 'trait'})
    self.assertEqual(['a', 'b', 'i', 's'], sorted(names))
    names = self.foo.copyable_trait_names(**{'type': lambda t: t in ('trait', 'property')})
    self.assertEqual(['a', 'b', 'i', 'p', 's'], sorted(names))