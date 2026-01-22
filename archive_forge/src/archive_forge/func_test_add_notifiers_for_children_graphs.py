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
def test_add_notifiers_for_children_graphs(self):
    observable1 = DummyObservable()
    child_observer1 = DummyObserver(observables=[observable1])
    observable2 = DummyObservable()
    child_observer2 = DummyObserver(observables=[observable2])
    parent_observer = DummyObserver(next_objects=[mock.Mock()])
    graph = ObserverGraph(node=parent_observer, children=[ObserverGraph(node=child_observer1), ObserverGraph(node=child_observer2)])
    call_add_or_remove_notifiers(graph=graph, remove=False)
    self.assertCountEqual(observable1.notifiers, [child_observer1.notifier])
    self.assertCountEqual(observable2.notifiers, [child_observer2.notifier])