import unittest
from unittest import mock
from traits.api import Bool, HasTraits, Int, Instance
from traits.observation._named_trait_observer import (
from traits.observation._observer_graph import ObserverGraph
from traits.observation._testing import (
def test_get_maintainer_excuse_old_value_with_no_notifiers(self):
    foo = ClassWithDefault()
    graph = create_graph(create_observer(name='instance', notify=True), create_observer(name='value1', notify=True))
    handler = mock.Mock()
    call_add_or_remove_notifiers(object=foo, graph=graph, handler=handler)
    try:
        foo.instance = ClassWithTwoValue()
    except Exception:
        self.fail('Reassigning the instance value should not fail.')