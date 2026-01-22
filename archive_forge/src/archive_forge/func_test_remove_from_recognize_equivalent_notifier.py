import unittest
from unittest import mock
import weakref
from traits.api import HasTraits, Instance, Int
from traits.observation._observer_change_notifier import ObserverChangeNotifier
from traits.observation._observer_graph import ObserverGraph
from traits.observation.exceptions import NotifierNotFound
def test_remove_from_recognize_equivalent_notifier(self):
    instance = DummyClass()
    handler = mock.Mock()
    observer_handler = mock.Mock()
    graph = mock.Mock()
    target = mock.Mock()
    notifier1 = create_notifier(handler=handler, observer_handler=observer_handler, graph=graph, target=target)
    notifier2 = create_notifier(handler=handler, observer_handler=observer_handler, graph=graph, target=target)
    notifier1.add_to(instance)
    notifier2.remove_from(instance)
    self.assertEqual(instance.notifiers, [])