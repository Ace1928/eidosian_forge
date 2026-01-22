import pickle
import unittest
from traits.api import Expression, HasTraits, Int, TraitError
def test_default_method_non_valid(self):

    class Foo(HasTraits):
        bar = Expression()

        def _bar_default(self):
            return '{x=y'
    f = Foo()
    msg = "The 'bar' trait of a Foo instance must be a valid"
    with self.assertRaisesRegex(TraitError, msg):
        f.bar