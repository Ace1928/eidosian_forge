import unittest
from unittest import mock
import weakref
from traits.api import HasTraits, Instance, Int
from traits.observation._observer_change_notifier import ObserverChangeNotifier
from traits.observation._observer_graph import ObserverGraph
from traits.observation.exceptions import NotifierNotFound
def observer_handler(event, graph, handler, target, dispatcher):
    old_notifiers = event.old._trait('value', 2)._notifiers(True)
    old_notifiers.remove(handler)
    new_notifiers = event.new._trait('value', 2)._notifiers(True)
    new_notifiers.append(handler)