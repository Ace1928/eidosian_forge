import pickle
import unittest
from traits.api import Expression, HasTraits, Int, TraitError
def test_default_static(self):

    class Foo(HasTraits):
        bar = Expression(default_value='1')
    f = Foo()
    self.assertEqual(f.bar, '1')
    self.assertEqual(eval(f.bar_), 1)