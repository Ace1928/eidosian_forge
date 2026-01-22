import unittest
from unittest import mock
from traits.has_traits import HasTraits
from traits.trait_types import Instance, Int
from traits.observation.api import (
from traits.observation.exceptions import NotifierNotFound
from traits.observation.expression import compile_expr, trait
from traits.observation.observe import (
from traits.observation._observer_graph import ObserverGraph
from traits.observation._testing import (
def test_apply_observers_different_dispatcher(self):
    self.dispatch_records = []

    def dispatcher(handler, event):
        self.dispatch_records.append((handler, event))
    foo = ClassWithNumber()
    handler = mock.Mock()
    apply_observers(object=foo, graphs=compile_expr(trait('number')), handler=handler, dispatcher=dispatcher)
    foo.number += 1
    self.assertEqual(len(self.dispatch_records), 1)