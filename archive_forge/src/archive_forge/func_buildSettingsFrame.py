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
def buildSettingsFrame(self, settings, ack=False):
    """
        Builds a single settings frame.
        """
    f = hyperframe.frame.SettingsFrame(0)
    if ack:
        f.flags.add('ACK')
    f.settings = settings
    return f