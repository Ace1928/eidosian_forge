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
def framesFromBytes(data):
    """
    Given a sequence of bytes, decodes them into frames.

    Note that this method should almost always be called only once, before
    making some assertions. This is because decoding HTTP/2 frames is extremely
    stateful, and this function doesn't preserve any of that state between
    calls.

    @param data: The serialized HTTP/2 frames.
    @type data: L{bytes}

    @returns: A list of HTTP/2 frames.
    @rtype: L{list} of L{hyperframe.frame.Frame} subclasses.
    """
    buffer = FrameBuffer()
    buffer.receiveData(data)
    return list(buffer)