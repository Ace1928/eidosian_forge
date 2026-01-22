import unittest
from traits.api import (
def test_trait_name_with_items(self):

    class Base(HasTraits):
        pass
    a = Base()
    a.add_trait('good_items', Str())
    self.assertNotIn('good_items', a.traits())