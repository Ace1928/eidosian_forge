import gc
import sys
import unittest
from traits.constants import DefaultValue
from traits.has_traits import (
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_errors import TraitError
from traits.trait_type import TraitType
from traits.trait_types import (
def test_modify_dict_in_list_with_new_value(self):
    instance = NestedContainerClass(list_of_dict=[{}])
    instance.list_of_dict.append(dict())
    try:
        instance.list_of_dict[-1]['key'] = 1
    except Exception:
        self.fail('Mutating a nested dict should not fail.')