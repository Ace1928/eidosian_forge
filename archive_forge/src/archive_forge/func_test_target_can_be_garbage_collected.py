import unittest
from unittest import mock
import weakref
from traits.api import HasTraits, Instance, Int
from traits.observation._observer_change_notifier import ObserverChangeNotifier
from traits.observation._observer_graph import ObserverGraph
from traits.observation.exceptions import NotifierNotFound
def test_target_can_be_garbage_collected(self):
    target = mock.Mock()
    target_ref = weakref.ref(target)
    notifier = create_notifier(target=target)
    del target
    self.assertIsNone(target_ref())