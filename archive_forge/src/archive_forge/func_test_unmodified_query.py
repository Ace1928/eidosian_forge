import unittest
from traits.api import (
def test_unmodified_query(self):
    names = self.foo.copyable_trait_names(**{'is_trait_type': lambda f: f(Str)})
    self.assertEqual(['s'], names)