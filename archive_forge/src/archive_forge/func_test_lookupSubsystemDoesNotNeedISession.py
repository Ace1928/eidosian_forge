import os
import signal
import struct
import sys
from unittest import skipIf
from zope.interface import implementer
from twisted.internet import defer, error, protocol
from twisted.internet.address import IPv4Address
from twisted.internet.error import ProcessDone, ProcessTerminated
from twisted.python import components, failure
from twisted.python.failure import Failure
from twisted.python.reflect import requireModule
from twisted.python.test.test_components import RegistryUsingMixin
from twisted.trial.unittest import TestCase
def test_lookupSubsystemDoesNotNeedISession(self):
    """
        Previously, if one only wanted to implement a subsystem, an ISession
        adapter wasn't needed because subsystems were looked up using the
        lookupSubsystem method on the avatar.
        """
    s = session.SSHSession(avatar=SubsystemOnlyAvatar(), conn=StubConnection())
    ret = s.request_subsystem(common.NS(b'subsystem') + b'data')
    self.assertTrue(ret)
    self.assertIsNotNone(s.client)
    self.assertIsNone(s.conn.closes.get(s))
    s.eofReceived()
    self.assertTrue(s.conn.closes.get(s))
    s.loseConnection()
    s.closed()