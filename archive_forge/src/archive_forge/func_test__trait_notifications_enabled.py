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
def test__trait_notifications_enabled(self):

    class Base(HasTraits):
        foo = Int(0)
        foo_notify_count = Int(0)

        def _foo_changed(self):
            self.foo_notify_count += 1
    a = Base()
    self.assertTrue(a._trait_notifications_enabled())
    old_count = a.foo_notify_count
    a.foo += 1
    self.assertEqual(a.foo_notify_count, old_count + 1)
    a._trait_change_notify(False)
    self.assertFalse(a._trait_notifications_enabled())
    old_count = a.foo_notify_count
    a.foo += 1
    self.assertEqual(a.foo_notify_count, old_count)
    a._trait_change_notify(True)
    self.assertTrue(a._trait_notifications_enabled())
    old_count = a.foo_notify_count
    a.foo += 1
    self.assertEqual(a.foo_notify_count, old_count + 1)