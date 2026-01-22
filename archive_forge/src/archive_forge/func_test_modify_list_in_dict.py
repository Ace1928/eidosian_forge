import gc
import sys
import unittest
from traits.constants import DefaultValue
from traits.has_traits import (
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_errors import TraitError
from traits.trait_type import TraitType
from traits.trait_types import (
def test_modify_list_in_dict(self):
    instance = NestedContainerClass(dict_of_list={'name': []})
    try:
        instance.dict_of_list['name'].append('word')
    except Exception:
        self.fail('Mutating a nested list should not fail.')