import copy
import operator
import pickle
import unittest.mock
from traits.api import HasTraits, Int, List
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_list_object import (
def test_dead_object_reference(self):
    foo = HasLengthConstrainedLists(at_most_five=[1, 2, 3, 4])
    list_object = foo.at_most_five
    del foo
    list_object.append(5)
    self.assertEqual(list_object, [1, 2, 3, 4, 5])
    with self.assertRaises(TraitError):
        list_object.append(4)