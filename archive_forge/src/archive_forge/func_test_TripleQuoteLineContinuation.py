import sys
import traceback
from typing import Optional
from twisted.conch import manhole
from twisted.conch.insults import insults
from twisted.conch.test.test_recvline import (
from twisted.internet import defer, error
from twisted.internet.testing import StringTransport
from twisted.trial import unittest
def test_TripleQuoteLineContinuation(self):
    """
        Evaluate line continuation in triple quotes.
        """
    done = self.recvlineClient.expect(b'done')
    self._testwrite(b"'''\n'''\ndone")

    def finished(ign):
        self._assertBuffer([b">>> '''", b"... '''", b"'\\n'", b'>>> done'])
    return done.addCallback(finished)