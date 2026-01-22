import copy
import pickle
import sys
import unittest
from unittest import mock
from traits.api import HasTraits
from traits.trait_dict_object import TraitDict, TraitDictEvent, TraitDictObject
from traits.trait_errors import TraitError
from traits.trait_types import Dict, Int, Str
def test_delitem_not_found(self):
    python_dict = dict()
    with self.assertRaises(KeyError) as python_e:
        del python_dict['x']
    td = TraitDict()
    with self.assertRaises(KeyError) as trait_e:
        del td['x']
    self.assertEqual(str(trait_e.exception), str(python_e.exception))