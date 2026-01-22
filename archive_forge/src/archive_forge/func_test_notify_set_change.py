import unittest
from unittest import mock
from traits.has_traits import HasTraits
from traits.observation._set_item_observer import SetItemObserver
from traits.observation._testing import (
from traits.trait_set_object import TraitSet
from traits.trait_types import Set
def test_notify_set_change(self):
    instance = ClassWithSet(values=set())
    graph = create_graph(create_observer(notify=True))
    handler = mock.Mock()
    call_add_or_remove_notifiers(object=instance.values, graph=graph, handler=handler)
    instance.values.add(1)
    ((event,), _), = handler.call_args_list
    self.assertEqual(event.added, set([1]))
    self.assertEqual(event.removed, set())