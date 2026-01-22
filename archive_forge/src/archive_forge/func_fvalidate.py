import unittest
from traits.api import HasStrictTraits, Int, TraitError
from traits.tests.tuple_test_mixin import TupleTestMixin
from traits.trait_types import ValidatedTuple
def fvalidate(x):
    if x == (5, 2):
        raise RuntimeError()
    return True