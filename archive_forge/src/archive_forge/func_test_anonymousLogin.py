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
def test_anonymousLogin(self):
    """
        Verify that a PB server using a portal configured with a checker which
        allows IAnonymous credentials can be logged into using IAnonymous
        credentials.
        """
    self.portal.registerChecker(checkers.AllowAnonymousAccess())
    d = self.clientFactory.login(credentials.Anonymous(), 'BRAINS!')

    def cbLoggedIn(perspective):
        return perspective.callRemote('echo', 123)
    d.addCallback(cbLoggedIn)
    d.addCallback(self.assertEqual, 123)
    self.establishClientAndServer()
    self.pump.flush()
    return d