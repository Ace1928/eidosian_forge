import gc
import sys
import unittest
from traits.constants import DefaultValue
from traits.has_traits import (
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_errors import TraitError
from traits.trait_type import TraitType
from traits.trait_types import (
def test_modify_set_in_list_with_new_value(self):
    instance = NestedContainerClass(list_of_set=[])
    instance.list_of_set.append(set())
    try:
        instance.list_of_set[0].add(1)
    except Exception:
        self.fail('Mutating a nested set should not fail.')