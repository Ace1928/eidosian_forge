import itertools
from zope.interface import directlyProvides, providedBy
from twisted.internet import defer, error, reactor, task
from twisted.internet.address import IPv4Address
from twisted.internet.testing import MemoryReactorClock, StringTransport
from twisted.python import failure
from twisted.python.compat import iterbytes
from twisted.test.test_internet import DummyProducer
from twisted.trial import unittest
from twisted.web import http
from twisted.web.test.test_http import (
def validateNotSent(*args):
    frames = framesFromBytes(b.value())
    self.assertEqual(len(frames), 2)
    self.assertFalse(isinstance(frames[-1], hyperframe.frame.DataFrame))
    a.resumeProducing()
    a.resumeProducing()
    a.resumeProducing()
    a.resumeProducing()
    a.resumeProducing()
    return cleanupCallback