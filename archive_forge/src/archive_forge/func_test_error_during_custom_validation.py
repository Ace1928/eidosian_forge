import unittest
from traits.api import HasStrictTraits, Int, TraitError
from traits.tests.tuple_test_mixin import TupleTestMixin
from traits.trait_types import ValidatedTuple
def test_error_during_custom_validation(self):

    def fvalidate(x):
        if x == (5, 2):
            raise RuntimeError()
        return True

    class Simple(HasStrictTraits):
        scalar_range = ValidatedTuple(Int(0), Int(1), fvalidate=fvalidate)
    simple = Simple()
    with self.assertRaises(RuntimeError):
        simple.scalar_range = (5, 2)