import copy
import pickle
import sys
import unittest
from unittest import mock
from traits.api import HasTraits
from traits.trait_dict_object import TraitDict, TraitDictEvent, TraitDictObject
from traits.trait_errors import TraitError
from traits.trait_types import Dict, Int, Str
def test_clear_empty_dictionary(self):
    td = TraitDict({}, key_validator=str_validator, value_validator=int_validator, notifiers=[self.notification_handler])
    td.clear()
    self.assertIsNone(self.added)
    self.assertIsNone(self.changed)
    self.assertIsNone(self.removed)