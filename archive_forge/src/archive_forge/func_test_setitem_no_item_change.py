import copy
import operator
import pickle
import unittest.mock
from traits.api import HasTraits, Int, List
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_list_object import (
def test_setitem_no_item_change(self):
    tl = TraitList([1, 2, 3], item_validator=int_item_validator, notifiers=[self.notification_handler])
    tl[0] = 1
    self.assertEqual(tl, [1, 2, 3])
    self.assertEqual(self.index, 0)
    self.assertEqual(self.removed, [1])
    self.assertEqual(self.added, [1])