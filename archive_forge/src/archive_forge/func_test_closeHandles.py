import errno
import gc
import gzip
import operator
import os
import signal
import stat
import sys
from unittest import SkipTest, skipIf
from io import BytesIO
from zope.interface.verify import verifyObject
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.python import procutils, runtime
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.python.log import msg
from twisted.trial import unittest
def test_closeHandles(self):
    """
        The win32 handles should be properly closed when the process exits.
        """
    import win32api
    connected = defer.Deferred()
    ended = defer.Deferred()

    class SimpleProtocol(protocol.ProcessProtocol):
        """
            A protocol that fires deferreds when connected and disconnected.
            """

        def makeConnection(self, transport):
            connected.callback(transport)

        def processEnded(self, reason):
            ended.callback(None)
    p = SimpleProtocol()
    pyArgs = [pyExe, b'-u', b'-c', b"print('hello')"]
    proc = reactor.spawnProcess(p, pyExe, pyArgs)

    def cbConnected(transport):
        self.assertIs(transport, proc)
        win32api.GetHandleInformation(proc.hProcess)
        win32api.GetHandleInformation(proc.hThread)
        self.hProcess = proc.hProcess
        self.hThread = proc.hThread
    connected.addCallback(cbConnected)

    def checkTerminated(ignored):
        self.assertIsNone(proc.pid)
        self.assertIsNone(proc.hProcess)
        self.assertIsNone(proc.hThread)
        self.assertRaises(win32api.error, win32api.GetHandleInformation, self.hProcess)
        self.assertRaises(win32api.error, win32api.GetHandleInformation, self.hThread)
    ended.addCallback(checkTerminated)
    return defer.gatherResults([connected, ended])