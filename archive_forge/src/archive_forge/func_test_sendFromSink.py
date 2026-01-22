from collections import OrderedDict
from twisted.trial import unittest
from twisted.words.xish import utility
from twisted.words.xish.domish import Element
from twisted.words.xish.utility import EventDispatcher
def test_sendFromSink(self):
    """
        Send an element from the sink and observe it from the source.
        """

    def cb(obj):
        called.append(obj)
    called = []
    self.pipe.source.addObserver('/test[@xmlns="testns"]', cb)
    element = Element(('testns', 'test'))
    self.pipe.sink.send(element)
    self.assertEqual([element], called)