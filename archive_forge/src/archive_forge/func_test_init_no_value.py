import copy
import operator
import pickle
import unittest.mock
from traits.api import HasTraits, Int, List
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_list_object import (
def test_init_no_value(self):
    tl = TraitList()
    self.assertEqual(tl, [])
    self.assertIs(tl.item_validator, _validate_everything)
    self.assertEqual(tl.notifiers, [])