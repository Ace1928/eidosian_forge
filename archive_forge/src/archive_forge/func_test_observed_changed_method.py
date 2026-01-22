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
def test_observed_changed_method(self):
    events = []

    class A(HasTraits):
        foo = Int()

        @observe('foo')
        def _foo_changed(self, event):
            events.append(event)
    a = A()
    a.foo = 23
    self.assertEqual(len(events), 1)
    event = events[0]
    self.assertEqual(event.object, a)
    self.assertEqual(event.name, 'foo')
    self.assertEqual(event.old, 0)
    self.assertEqual(event.new, 23)