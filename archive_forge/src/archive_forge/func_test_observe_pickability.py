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
def test_observe_pickability(self):
    person = PersonWithObserve()
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        serialized = pickle.dumps(person, protocol=protocol)
        deserialized = pickle.loads(serialized)
        deserialized.age += 1
        self.assertEqual(len(deserialized.events), 1)