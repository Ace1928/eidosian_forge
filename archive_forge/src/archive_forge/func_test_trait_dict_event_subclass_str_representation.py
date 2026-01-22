import copy
import pickle
import sys
import unittest
from unittest import mock
from traits.api import HasTraits
from traits.trait_dict_object import TraitDict, TraitDictEvent, TraitDictObject
from traits.trait_errors import TraitError
from traits.trait_types import Dict, Int, Str
def test_trait_dict_event_subclass_str_representation(self):
    """ Test string representation of a subclass of the TraitDictEvent
        class. """

    class DifferentName(TraitDictEvent):
        pass
    desired_repr = 'DifferentName(removed={}, added={}, changed={})'
    differnt_name_subclass = DifferentName()
    self.assertEqual(desired_repr, str(differnt_name_subclass))
    self.assertEqual(desired_repr, repr(differnt_name_subclass))