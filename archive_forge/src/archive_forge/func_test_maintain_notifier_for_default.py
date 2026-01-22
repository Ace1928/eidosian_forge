import unittest
from unittest import mock
from traits.api import Bool, HasTraits, Int, Instance
from traits.observation._named_trait_observer import (
from traits.observation._observer_graph import ObserverGraph
from traits.observation._testing import (
def test_maintain_notifier_for_default(self):
    foo = ClassWithDefault()
    graph = create_graph(create_observer(name='instance', notify=True), create_observer(name='value1', notify=True))
    handler = mock.Mock()
    call_add_or_remove_notifiers(object=foo, graph=graph, handler=handler)
    self.assertNotIn('instance', foo.__dict__)
    foo.instance
    self.assertEqual(handler.call_count, 0)
    foo.instance.value1 += 1
    self.assertEqual(handler.call_count, 1)