import sys
import traceback
from typing import Optional
from twisted.conch import manhole
from twisted.conch.insults import insults
from twisted.conch.test.test_recvline import (
from twisted.internet import defer, error
from twisted.internet.testing import StringTransport
from twisted.trial import unittest
def test_controlA(self):
    """
        CTRL-A can be used as HOME - returning cursor to beginning of
        current line buffer.
        """
    self._testwrite(b'rint "hello"' + b'\x01' + b'p')
    d = self.recvlineClient.expect(b'print "hello"')

    def cb(ignore):
        self._assertBuffer([b'>>> print "hello"'])
    return d.addCallback(cb)