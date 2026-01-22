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
def test_decorated_changed_method(self):
    events = []

    class A(HasTraits):
        foo = Int()

        @on_trait_change('foo')
        def _foo_changed(self, obj, name, old, new):
            events.append((obj, name, old, new))
    a = A()
    a.foo = 23
    self.assertEqual(events, [(a, 'foo', 0, 23)])