import gc
import sys
import unittest
from traits.constants import DefaultValue
from traits.has_traits import (
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_errors import TraitError
from traits.trait_type import TraitType
from traits.trait_types import (
@requires_numpy
def test_exception_from_numpy_comparison_ignored(self):

    class MultiArrayDataSource(HasTraits):
        data = Either(None, Array)
    b = MultiArrayDataSource(data=numpy.array([1, 2]))
    round(3.14159, 2)