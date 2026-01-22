import gc
import sys
import unittest
from traits.constants import DefaultValue
from traits.has_traits import (
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_errors import TraitError
from traits.trait_type import TraitType
from traits.trait_types import (
def test_modify_dict_in_dict_no_events(self):
    instance = NestedContainerClass(dict_of_dict={'1': {'2': 2}})
    instance.on_trait_change(self.change_handler, 'dict_of_dict_items')
    instance.dict_of_dict['1']['3'] = 3
    self.assertEqual(len(self.events), 0, 'Expected no events.')