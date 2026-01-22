import gc
import os
import sys
import time
import weakref
from collections import deque
from io import BytesIO as StringIO
from typing import Dict
from zope.interface import Interface, implementer
from twisted.cred import checkers, credentials, portal
from twisted.cred.error import UnauthorizedLogin, UnhandledCredentials
from twisted.internet import address, main, protocol, reactor
from twisted.internet.defer import Deferred, gatherResults, succeed
from twisted.internet.error import ConnectionRefusedError
from twisted.internet.testing import _FakeConnector
from twisted.protocols.policies import WrappingFactory
from twisted.python import failure, log
from twisted.python.compat import iterbytes
from twisted.spread import jelly, pb, publish, util
from twisted.trial import unittest
def test_tooManyRefs(self):
    l = []
    e = []
    c, s, pump = connectedServerAndClient(test=self)
    foo = NestedRemote()
    s.setNameForLocal('foo', foo)
    x = c.remoteForName('foo')
    for igno in range(pb.MAX_BROKER_REFS + 10):
        if s.transport.closed or c.transport.closed:
            break
        x.callRemote('getSimple').addCallbacks(l.append, e.append)
        pump.pump()
    expected = pb.MAX_BROKER_REFS - 1
    self.assertTrue(s.transport.closed, 'transport was not closed')
    self.assertEqual(len(l), expected, f'expected {expected} got {len(l)}')