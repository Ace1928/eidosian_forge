import unittest
from traits.api import (
def test_trait_set_replaced_and_check(self):
    b = Foo()
    b.add_trait('num', Str())
    b.num = 'first'
    self.assertEqual(b.num, 'first')
    self.assertEqual(b.trait('num'), b.traits()['num'])