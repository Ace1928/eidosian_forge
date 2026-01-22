import datetime
import decimal
from typing import ClassVar, Dict, Type, TypeVar
from unittest import skipIf
from zope.interface import implementer
from zope.interface.verify import verifyClass, verifyObject
from twisted.internet import address, defer, error, interfaces, protocol, reactor
from twisted.internet.testing import StringTransport
from twisted.protocols import amp
from twisted.python import filepath
from twisted.python.failure import Failure
from twisted.test import iosim
from twisted.trial.unittest import TestCase
def test_protocolSwitch(self, switcher=SimpleSymmetricCommandProtocol, spuriousTraffic=False, spuriousError=False):
    """
        Verify that it is possible to switch to another protocol mid-connection and
        send data to it successfully.
        """
    self.testSucceeded = False
    serverDeferred = defer.Deferred()
    serverProto = switcher(serverDeferred)
    clientDeferred = defer.Deferred()
    clientProto = switcher(clientDeferred)
    c, s, p = connectedServerAndClient(ServerClass=lambda: serverProto, ClientClass=lambda: clientProto)
    if spuriousTraffic:
        wfdr = []
        c.callRemote(WaitForever).addErrback(wfdr.append)
    switchDeferred = c.switchToTestProtocol()
    if spuriousTraffic:
        self.assertRaises(amp.ProtocolSwitched, c.sendHello, b'world')

    def cbConnsLost(info):
        (serverSuccess, serverData), (clientSuccess, clientData) = info
        self.assertTrue(serverSuccess)
        self.assertTrue(clientSuccess)
        self.assertEqual(b''.join(serverData), SWITCH_CLIENT_DATA)
        self.assertEqual(b''.join(clientData), SWITCH_SERVER_DATA)
        self.testSucceeded = True

    def cbSwitch(proto):
        return defer.DeferredList([serverDeferred, clientDeferred]).addCallback(cbConnsLost)
    switchDeferred.addCallback(cbSwitch)
    p.flush()
    if serverProto.maybeLater is not None:
        serverProto.maybeLater.callback(serverProto.maybeLaterProto)
        p.flush()
    if spuriousTraffic:
        if spuriousError:
            s.waiting.errback(amp.RemoteAmpError(b'SPURIOUS', "Here's some traffic in the form of an error."))
        else:
            s.waiting.callback({})
        p.flush()
    c.transport.loseConnection()
    p.flush()
    self.assertTrue(self.testSucceeded)