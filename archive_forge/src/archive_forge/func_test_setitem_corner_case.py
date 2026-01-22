import copy
import operator
import pickle
import unittest.mock
from traits.api import HasTraits, Int, List
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_list_object import (
def test_setitem_corner_case(self):
    tl = TraitList(range(7), notifiers=[self.notification_handler])
    tl[5:2] = [10, 11, 12]
    self.assertEqual(tl, [0, 1, 2, 3, 4, 10, 11, 12, 5, 6])
    self.assertEqual(self.index, 5)
    self.assertEqual(self.removed, [])
    self.assertEqual(self.added, [10, 11, 12])