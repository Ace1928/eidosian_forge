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
def test_newStyle(self):
    """
        Create a new style object, send it over the wire, and check the result.
        """
    orig = NewStyleCopy('value')
    d = self.ref.callRemote('echo', orig)
    self.pump.flush()

    def cb(res):
        self.assertIsInstance(res, NewStyleCopy)
        self.assertEqual(res.s, 'value')
        self.assertFalse(res is orig)
    d.addCallback(cb)
    return d