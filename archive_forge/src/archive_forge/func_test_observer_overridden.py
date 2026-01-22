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
def test_observer_overridden(self):

    class BaseClass(HasTraits):
        events = List()

        @observe('value')
        def handler(self, event):
            self.events.append(event)

    class SubclassOverriden(BaseClass):
        value = Int()
        handler = None
    instance = SubclassOverriden()
    instance.value += 1
    self.assertEqual(len(instance.events), 0)