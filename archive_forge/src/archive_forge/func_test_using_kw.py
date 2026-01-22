import unittest
from traits.api import HasTraits, Instance, Int
def test_using_kw(self):
    bar = Bar(b=5)
    foo = Foo(bar=bar)
    self.assertEqual(foo.bar.b, 5)