import gc
import sys
import unittest
from traits.constants import DefaultValue
from traits.has_traits import (
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_errors import TraitError
from traits.trait_type import TraitType
from traits.trait_types import (
def test_validate_exception_propagates(self):

    class A(HasTraits):
        foo = RaisingValidator()
        bar = Either(None, RaisingValidator())
    a = A()
    with self.assertRaises(ZeroDivisionError):
        a.foo = 'foo'
    with self.assertRaises(ZeroDivisionError):
        a.bar = 'foo'