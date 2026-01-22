import copy
import pickle
import unittest
from traits.has_traits import (
from traits.ctrait import CTrait
from traits.observation.api import (
from traits.observation.exception_handling import (
from traits.traits import ForwardProperty, generic_trait
from traits.trait_types import Event, Float, Instance, Int, List, Map, Str
from traits.trait_errors import TraitError
def test_add_listener_and_remove_listener_deprecated(self):

    def listener(cls):
        pass
    with self.assertWarnsRegex(DeprecationWarning, 'add_listener is deprecated'):
        MetaHasTraits.add_listener(listener)
    with self.assertWarnsRegex(DeprecationWarning, 'remove_listener is deprecated'):
        MetaHasTraits.remove_listener(listener)