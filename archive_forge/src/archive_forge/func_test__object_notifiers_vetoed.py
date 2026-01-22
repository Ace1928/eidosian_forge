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
def test__object_notifiers_vetoed(self):

    class SomeEvent(HasTraits):
        event_id = Int()

    class Target(HasTraits):
        event = Event(Instance(SomeEvent))
        event_count = Int(0)
    target = Target()
    event = SomeEvent(event_id=9)

    def object_handler(object, name, old, new):
        if name == 'event':
            object.event_count += 1
    target.on_trait_change(object_handler, name='anytrait')
    self.assertFalse(event._trait_notifications_vetoed())
    old_count = target.event_count
    target.event = event
    self.assertEqual(target.event_count, old_count + 1)
    event._trait_veto_notify(True)
    self.assertTrue(event._trait_notifications_vetoed())
    old_count = target.event_count
    target.event = event
    self.assertEqual(target.event_count, old_count)
    event._trait_veto_notify(False)
    self.assertFalse(event._trait_notifications_vetoed())
    old_count = target.event_count
    target.event = event
    self.assertEqual(target.event_count, old_count + 1)