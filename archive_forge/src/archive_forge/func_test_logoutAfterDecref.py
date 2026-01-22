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
def test_logoutAfterDecref(self):
    """
        If a L{RemoteReference} to an L{IPerspective} avatar is decrefed and
        there remain no other references to the avatar on the server, the
        avatar is garbage collected and the logout method called.
        """
    loggedOut = Deferred()

    class EventPerspective(pb.Avatar):
        """
            An avatar which fires a Deferred when it is logged out.
            """

        def __init__(self, avatarId):
            pass

        def logout(self):
            loggedOut.callback(None)
    self.realm.perspectiveFactory = EventPerspective
    self.portal.registerChecker(checkers.InMemoryUsernamePasswordDatabaseDontUse(foo=b'bar'))
    d = self.clientFactory.login(credentials.UsernamePassword(b'foo', b'bar'), 'BRAINS!')

    def cbLoggedIn(avatar):
        return loggedOut
    d.addCallback(cbLoggedIn)

    def cbLoggedOut(ignored):
        self.assertEqual(self.serverFactory.protocolInstance._localCleanup, {})
    d.addCallback(cbLoggedOut)
    self.establishClientAndServer()
    self.pump.flush()
    gc.collect()
    self.pump.flush()
    return d