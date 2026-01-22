from collections import OrderedDict
from twisted.trial import unittest
from twisted.words.xish import utility
from twisted.words.xish.domish import Element
from twisted.words.xish.utility import EventDispatcher
def test_cleanUpRemoveXPathObserver(self):
    """
        Test observer clean-up after removeObserver for XPath events.
        """
    d = EventDispatcher()
    cb = CallbackTracker()
    msg = Element((None, 'message'))
    d.addObserver('/message', cb.call)
    d.dispatch(msg)
    self.assertEqual(1, cb.called)
    d.removeObserver('/message', cb.call)
    self.assertEqual(0, len(d._xpathObservers.pop(0)))