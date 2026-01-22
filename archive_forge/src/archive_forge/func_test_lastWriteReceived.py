import itertools
import os
import sys
from unittest import skipIf
from twisted.internet import defer, error, protocol, reactor, stdio
from twisted.python import filepath, log
from twisted.python.reflect import requireModule
from twisted.python.runtime import platform
from twisted.test.test_tcp import ConnectionLostNotifyingProtocol
from twisted.trial.unittest import SkipTest, TestCase
def test_lastWriteReceived(self):
    """
        Verify that a write made directly to stdout using L{os.write}
        after StandardIO has finished is reliably received by the
        process reading that stdout.
        """
    p = StandardIOTestProcessProtocol()
    try:
        self._spawnProcess(p, b'stdio_test_lastwrite', UNIQUE_LAST_WRITE_STRING, usePTY=True)
    except ValueError as e:
        raise SkipTest(str(e))

    def processEnded(reason):
        """
            Asserts that the parent received the bytes written by the child
            immediately after the child starts.
            """
        self.assertTrue(p.data[1].endswith(UNIQUE_LAST_WRITE_STRING), f'Received {p.data!r} from child, did not find expected bytes.')
        reason.trap(error.ProcessDone)
    return self._requireFailure(p.onCompletion, processEnded)