import gc
import sys
import unittest
from traits.constants import DefaultValue
from traits.has_traits import (
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_errors import TraitError
from traits.trait_type import TraitType
from traits.trait_types import (
def test_init_list_depends(self):
    """ Using two lists with bracket notation in extended name notation
        should not raise an error.
        """
    list_test = ListUpdatesTest()
    list_test.a.append(0)
    list_test.b = [1, 2, 3]
    list_test.b[0] = 0
    self.assertEqual(list_test.events_received, 3)