import unittest
from traits.api import HasTraits, Instance, Int
def test_not_using_kw(self):
    foo = Foo()
    self.assertEqual(foo.bar, None)