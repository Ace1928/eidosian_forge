import copy
import operator
import pickle
import unittest.mock
from traits.api import HasTraits, Int, List
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_list_object import (
def test_imul_no_notification_for_empty_list(self):
    for multiplier in [-1, 0, 1, 2]:
        with self.subTest(multiplier=multiplier):
            tl = TraitList([], item_validator=int_item_validator, notifiers=[self.notification_handler])
            tl *= multiplier
            self.assertEqual(tl, [])
            self.assertIsNone(self.index)
            self.assertIsNone(self.removed)
            self.assertIsNone(self.added)