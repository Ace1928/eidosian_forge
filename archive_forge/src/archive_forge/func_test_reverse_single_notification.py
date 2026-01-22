import copy
import operator
import pickle
import unittest.mock
from traits.api import HasTraits, Int, List
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_list_object import (
def test_reverse_single_notification(self):
    notifier = unittest.mock.Mock()
    tl = TraitList([1, 2, 3, 4, 5], notifiers=[notifier])
    notifier.assert_not_called()
    tl.reverse()
    self.assertEqual(notifier.call_count, 1)