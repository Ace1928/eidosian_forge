import copy
import operator
import pickle
import unittest.mock
from traits.api import HasTraits, Int, List
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_list_object import (
def test_delitem_negative_step(self):
    tl = TraitList([1, 2, 3, 4, 5], item_validator=int_item_validator, notifiers=[self.notification_handler])
    del tl[::-2]
    self.assertEqual(tl, [2, 4])
    self.assertEqual(self.index, slice(0, 5, 2))
    self.assertEqual(self.removed, [1, 3, 5])
    self.assertEqual(self.added, [])