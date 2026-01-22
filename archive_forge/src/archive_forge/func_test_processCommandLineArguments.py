import io
import os
import signal
import subprocess
import sys
import threading
from unittest import skipIf
import hamcrest
from twisted.internet import utils
from twisted.internet.defer import Deferred, inlineCallbacks, succeed
from twisted.internet.error import ProcessDone, ProcessTerminated
from twisted.internet.interfaces import IProcessTransport, IReactorProcess
from twisted.internet.protocol import ProcessProtocol
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath, _asFilesystemBytes
from twisted.python.log import err, msg
from twisted.python.runtime import platform
from twisted.test.test_process import Accumulator
from twisted.trial.unittest import SynchronousTestCase, TestCase
import sys
from twisted.internet import process
def test_processCommandLineArguments(self):
    """
        Arguments given to spawnProcess are passed to the child process as
        originally intended.
        """
    us = b'twisted.internet.test.process_cli'
    args = [b'hello', b'"', b' \t|<>^&', b'"\\\\"hello\\\\"', b'"foo\\ bar baz\\""']
    allChars = ''.join(map(chr, range(1, 255)))
    if isinstance(allChars, str):
        allChars.encode('utf-8')
    reactor = self.buildReactor()

    def processFinished(finishedArgs):
        output, err, code = finishedArgs
        output = output.split(b'\x00')
        output.pop()
        self.assertEqual(args, output)

    def shutdown(result):
        reactor.stop()
        return result

    def spawnChild():
        d = succeed(None)
        d.addCallback(lambda dummy: utils.getProcessOutputAndValue(pyExe, [b'-m', us] + args, env=properEnv, reactor=reactor))
        d.addCallback(processFinished)
        d.addBoth(shutdown)
    reactor.callWhenRunning(spawnChild)
    self.runReactor(reactor)