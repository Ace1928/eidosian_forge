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
def test_newStyleCache(self):
    """
        A new-style cacheable object can be retrieved and re-retrieved over a
        single connection.  The value of an attribute of the cacheable can be
        accessed on the receiving side.
        """
    d = self.ref.callRemote('giveMeCache', self.orig)

    def cb(res, again):
        self.assertIsInstance(res, NewStyleCacheCopy)
        self.assertEqual('value', res.s)
        self.assertIsNot(self.orig, res)
        if again:
            self.res = res
            return self.ref.callRemote('giveMeCache', self.orig)
    d.addCallback(cb, True)
    d.addCallback(cb, False)
    return d