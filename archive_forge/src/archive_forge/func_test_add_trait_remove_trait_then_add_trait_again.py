import unittest
from unittest import mock
from traits.api import Bool, HasTraits, Int, Instance
from traits.observation._named_trait_observer import (
from traits.observation._observer_graph import ObserverGraph
from traits.observation._testing import (
def test_add_trait_remove_trait_then_add_trait_again(self):
    graph = create_graph(create_observer(name='new_value', notify=True, optional=True))
    handler = mock.Mock()
    foo = ClassWithInstance()
    call_add_or_remove_notifiers(object=foo, graph=graph, handler=handler, remove=False)
    foo.add_trait('new_value', Int())
    foo.new_value += 1
    handler.assert_called_once()
    handler.reset_mock()
    foo.remove_trait('new_value')
    foo.add_trait('new_value', Int())
    foo.new_value += 1
    handler.assert_called_once()