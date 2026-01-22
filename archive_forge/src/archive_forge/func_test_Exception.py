import sys
import traceback
from typing import Optional
from twisted.conch import manhole
from twisted.conch.insults import insults
from twisted.conch.test.test_recvline import (
from twisted.internet import defer, error
from twisted.internet.testing import StringTransport
from twisted.trial import unittest
def test_Exception(self):
    """
        Evaluate raising an exception.
        """
    done = self.recvlineClient.expect(b'done')
    self._testwrite(b"raise Exception('foo bar baz')\ndone")

    def finished(ign):
        self._assertBuffer([b">>> raise Exception('foo bar baz')", b'Traceback (most recent call last):', b'  File "<console>", line 1, in ' + defaultFunctionName.encode('utf-8'), b'Exception: foo bar baz', b'>>> done'])
    done.addCallback(finished)
    return done