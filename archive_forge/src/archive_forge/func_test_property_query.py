import unittest
from traits.api import (
def test_property_query(self):
    names = self.foo.copyable_trait_names(**{'property_fields': lambda p: p and p[1].__name__ == '_set_p'})
    self.assertEqual(['p'], names)