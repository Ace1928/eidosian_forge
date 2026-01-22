import errno
import os
import socket
from unittest import skipIf
from twisted.internet import interfaces, reactor
from twisted.internet.defer import gatherResults, maybeDeferred
from twisted.internet.protocol import Protocol, ServerFactory
from twisted.internet.tcp import (
from twisted.python import log
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
@skipIf(platform.getType() == 'win32', 'Windows accept(2) cannot generate EPERM')
def test_permissionFailure(self):
    """
        C{accept(2)} returning C{EPERM} is treated as a transient
        failure and the call retried no more than the maximum number
        of consecutive C{accept(2)} calls.
        """
    maximumNumberOfAccepts = 123
    acceptCalls = [0]

    class FakeSocketWithAcceptLimit:
        """
            Pretend to be a socket in an overloaded system whose
            C{accept} method can only be called
            C{maximumNumberOfAccepts} times.
            """

        def accept(oself):
            acceptCalls[0] += 1
            if acceptCalls[0] > maximumNumberOfAccepts:
                self.fail('Maximum number of accept calls exceeded.')
            raise OSError(EPERM, os.strerror(EPERM))
    for _ in range(maximumNumberOfAccepts):
        self.assertRaises(socket.error, FakeSocketWithAcceptLimit().accept)
    self.assertRaises(self.failureException, FakeSocketWithAcceptLimit().accept)
    acceptCalls = [0]
    factory = ServerFactory()
    port = self.port(0, factory, interface='127.0.0.1')
    port.numberAccepts = 123
    self.patch(port, 'socket', FakeSocketWithAcceptLimit())
    port.doRead()
    self.assertEquals(port.numberAccepts, 1)