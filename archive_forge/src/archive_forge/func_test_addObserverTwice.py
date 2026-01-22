from collections import OrderedDict
from twisted.trial import unittest
from twisted.words.xish import utility
from twisted.words.xish.domish import Element
from twisted.words.xish.utility import EventDispatcher
def test_addObserverTwice(self):
    """
        Test adding two observers for the same query.

        When the event is dispatched both of the observers need to be called.
        """
    d = EventDispatcher()
    cb1 = CallbackTracker()
    cb2 = CallbackTracker()
    d.addObserver('//event/testevent', cb1.call)
    d.addObserver('//event/testevent', cb2.call)
    d.dispatch(d, '//event/testevent')
    self.assertEqual(cb1.called, 1)
    self.assertEqual(cb1.obj, d)
    self.assertEqual(cb2.called, 1)
    self.assertEqual(cb2.obj, d)