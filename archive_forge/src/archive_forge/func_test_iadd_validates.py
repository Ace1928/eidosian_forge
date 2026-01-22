import copy
import operator
import pickle
import unittest.mock
from traits.api import HasTraits, Int, List
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_list_object import (
def test_iadd_validates(self):
    tl = TraitList([4, 5], item_validator=int_item_validator, notifiers=[self.notification_handler])
    with self.assertRaises(TraitError):
        tl += [6, 7, 8.0]
    self.assertEqual(tl, [4, 5])
    self.assertIsNone(self.index)
    self.assertIsNone(self.removed)
    self.assertIsNone(self.added)