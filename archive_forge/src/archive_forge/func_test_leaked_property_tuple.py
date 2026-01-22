import gc
import sys
import unittest
from traits.constants import DefaultValue
from traits.has_traits import (
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_errors import TraitError
from traits.trait_type import TraitType
from traits.trait_types import (
def test_leaked_property_tuple(self):
    """ the property ctrait constructor shouldn't leak a tuple. """

    class A(HasTraits):
        prop = Property()
    a = A()
    self.assertEqual(sys.getrefcount(a.trait('prop').property_fields), 1)