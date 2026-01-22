import unittest
from traits.api import (
def test_trait_set_bad(self):
    b = Foo(num=23)
    with self.assertRaises(TraitError):
        b.num = 'first'
    self.assertEqual(b.num, 23)