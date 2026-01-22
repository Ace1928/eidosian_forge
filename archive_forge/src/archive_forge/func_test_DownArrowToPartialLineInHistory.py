import os
import sys
from unittest import skipIf
from twisted.conch import recvline
from twisted.conch.insults import insults
from twisted.cred import portal
from twisted.internet import defer, error
from twisted.internet.testing import StringTransport
from twisted.python import components, filepath, reflect
from twisted.python.compat import iterbytes
from twisted.python.reflect import requireModule
from twisted.trial.unittest import SkipTest, TestCase
from twisted.conch import telnet
from twisted.conch.insults import helper
from twisted.conch.test.loopback import LoopbackRelay
from twisted.cred import checkers
from twisted.conch.test import test_telnet
def test_DownArrowToPartialLineInHistory(self):
    """
        Pressing down arrow to visit an entry that was added to the
        history by pressing the up arrow instead of return does not
        raise a L{TypeError}.

        @see: U{http://twistedmatrix.com/trac/ticket/9031}

        @return: A L{defer.Deferred} that fires when C{b"done"} is
            echoed back.
        """
    return self._trivialTest(b'first line\n' + b'partial line' + up + down + b'\ndone', [b'>>> first line', b'first line', b'>>> partial line', b'partial line', b'>>> done'])