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
def test_inheritableConflict(self) -> None:
    """
        If our file descriptor mapping requests that file descriptors change
        places, we must DUP2 them to a new location before DUP2ing them back.
        """
    self.assertEqual(_getFileActions([(0, False), (1, False)], {0: 1, 1: 0}, CLOSE, DUP2), [(DUP2, 0, 2), (DUP2, 1, 0), (DUP2, 2, 1), (CLOSE, 2)])