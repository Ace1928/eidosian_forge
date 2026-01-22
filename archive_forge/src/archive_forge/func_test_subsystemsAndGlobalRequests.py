import struct
from itertools import chain
from typing import Dict, List, Tuple
from twisted.conch.test.keydata import (
from twisted.conch.test.loopback import LoopbackRelay
from twisted.cred import portal
from twisted.cred.error import UnauthorizedLogin
from twisted.internet import defer, protocol, reactor
from twisted.internet.error import ProcessTerminated
from twisted.python import failure, log
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.python import components
def test_subsystemsAndGlobalRequests(self):
    """
        Run the Conch server against the Conch client.  Set up several different
        channels which exercise different behaviors and wait for them to
        complete.  Verify that the channels with errors log them.
        """
    channel = self._ourServerOurClientTest()

    def cbSubsystem(channel):
        self.channel = channel
        return self.assertFailure(channel.conn.sendRequest(channel, b'subsystem', common.NS(b'not-crazy'), 1), Exception)
    channel.addCallback(cbSubsystem)

    def cbNotCrazyFailed(ignored):
        channel = self.channel
        return channel.conn.sendRequest(channel, b'subsystem', common.NS(b'crazy'), 1)
    channel.addCallback(cbNotCrazyFailed)

    def cbGlobalRequests(ignored):
        channel = self.channel
        d1 = channel.conn.sendGlobalRequest(b'foo', b'bar', 1)
        d2 = channel.conn.sendGlobalRequest(b'foo-2', b'bar2', 1)
        d2.addCallback(self.assertEqual, b'data')
        d3 = self.assertFailure(channel.conn.sendGlobalRequest(b'bar', b'foo', 1), Exception)
        return defer.gatherResults([d1, d2, d3])
    channel.addCallback(cbGlobalRequests)

    def disconnect(ignored):
        self.assertEqual(self.realm.avatar.globalRequests, {'foo': b'bar', 'foo_2': b'bar2'})
        channel = self.channel
        channel.conn.transport.expectedLoseConnection = True
        channel.conn.serviceStopped()
        channel.loseConnection()
    channel.addCallback(disconnect)
    return channel