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
def test_add_trait_notifiers_notify_flag_is_false(self):
    observable = DummyObservable()
    notifier = DummyNotifier()
    observer = DummyObserver(notify=False, observables=[observable], notifier=notifier)
    graph = ObserverGraph(node=observer)
    call_add_or_remove_notifiers(graph=graph, remove=False)
    self.assertEqual(observable.notifiers, [])