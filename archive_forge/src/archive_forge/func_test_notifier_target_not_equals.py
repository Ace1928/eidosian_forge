import unittest
from unittest import mock
import weakref
from traits.api import HasTraits, Instance, Int
from traits.observation._observer_change_notifier import ObserverChangeNotifier
from traits.observation._observer_graph import ObserverGraph
from traits.observation.exceptions import NotifierNotFound
def test_notifier_target_not_equals(self):
    observer_handler = mock.Mock()
    handler = mock.Mock()
    graph = mock.Mock()
    target1 = mock.Mock()
    target2 = mock.Mock()
    notifier1 = create_notifier(observer_handler=observer_handler, handler=handler, graph=graph, target=target1, dispatcher=dispatch_here)
    notifier2 = create_notifier(observer_handler=observer_handler, handler=handler, graph=graph, target=target2, dispatcher=dispatch_here)
    self.assertFalse(notifier1.equals(notifier2), 'Expected notifier1 to see notifier2 as different.')
    self.assertFalse(notifier2.equals(notifier1), 'Expected notifier2 to see notifier1 as different.')