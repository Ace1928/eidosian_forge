import copy
import operator
import pickle
import unittest.mock
from traits.api import HasTraits, Int, List
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_list_object import (
def test_setitem_iterable(self):
    tl = TraitList([1, 2, 3], item_validator=int_item_validator, notifiers=[self.notification_handler])
    tl[:] = (x ** 2 for x in range(4))
    self.assertEqual(tl, [0, 1, 4, 9])
    self.assertEqual(self.index, 0)
    self.assertEqual(self.removed, [1, 2, 3])
    self.assertEqual(self.added, [0, 1, 4, 9])