from collections import OrderedDict
from twisted.trial import unittest
from twisted.words.xish import utility
from twisted.words.xish.domish import Element
from twisted.words.xish.utility import EventDispatcher
def testStuff(self):
    d = EventDispatcher()
    cb1 = CallbackTracker()
    cb2 = CallbackTracker()
    cb3 = CallbackTracker()
    d.addObserver('/message/body', cb1.call)
    d.addObserver('/message', cb1.call)
    d.addObserver('/presence', cb2.call)
    d.addObserver('//event/testevent', cb3.call)
    msg = Element(('ns', 'message'))
    msg.addElement('body')
    pres = Element(('ns', 'presence'))
    pres.addElement('presence')
    d.dispatch(msg)
    self.assertEqual(cb1.called, 2)
    self.assertEqual(cb1.obj, msg)
    self.assertEqual(cb2.called, 0)
    d.dispatch(pres)
    self.assertEqual(cb1.called, 2)
    self.assertEqual(cb2.called, 1)
    self.assertEqual(cb2.obj, pres)
    self.assertEqual(cb3.called, 0)
    d.dispatch(d, '//event/testevent')
    self.assertEqual(cb3.called, 1)
    self.assertEqual(cb3.obj, d)
    d.removeObserver('/presence', cb2.call)
    d.dispatch(pres)
    self.assertEqual(cb2.called, 1)