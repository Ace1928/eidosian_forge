import unittest
from unittest import mock
import weakref
from traits.api import HasTraits, Instance, Int
from traits.observation._observer_change_notifier import ObserverChangeNotifier
from traits.observation._observer_graph import ObserverGraph
from traits.observation.exceptions import NotifierNotFound
def test_basic_instance_change(self):

    class Bar(HasTraits):
        value = Int()

    class Foo(HasTraits):
        bar = Instance(Bar)
    on_bar_value_changed = self.handler
    bar = Bar()
    foo1 = Foo(bar=bar)
    foo2 = Foo(bar=bar)

    def observer_handler(event, graph, handler, target, dispatcher):
        old_notifiers = event.old._trait('value', 2)._notifiers(True)
        old_notifiers.remove(handler)
        new_notifiers = event.new._trait('value', 2)._notifiers(True)
        new_notifiers.append(handler)
    bar._trait('value', 2)._notifiers(True).extend([on_bar_value_changed, on_bar_value_changed])
    notifier_foo1 = create_notifier(observer_handler=observer_handler, event_factory=self.event_factory, graph=ObserverGraph(node=None), handler=on_bar_value_changed, target=foo1, dispatcher=dispatch_here)
    notifier_foo1.add_to(foo1._trait('bar', 2))
    notifier_foo2 = create_notifier(observer_handler=observer_handler, event_factory=self.event_factory, graph=ObserverGraph(node=None), handler=on_bar_value_changed, target=foo2, dispatcher=dispatch_here)
    notifier_foo2.add_to(foo2._trait('bar', 2))
    self.event_args_list.clear()
    new_bar = Bar(value=1)
    foo1.bar = new_bar
    foo2.bar = new_bar
    new_bar.value += 1
    self.assertEqual(len(self.event_args_list), 2)
    self.event_args_list.clear()
    bar.value += 1
    self.assertEqual(len(self.event_args_list), 0)