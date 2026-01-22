import sys
import traceback
from typing import Optional
from twisted.conch import manhole
from twisted.conch.insults import insults
from twisted.conch.test.test_recvline import (
from twisted.internet import defer, error
from twisted.internet.testing import StringTransport
from twisted.trial import unittest
def test_ClassDefinition(self):
    """
        Evaluate class definition.
        """
    done = self.recvlineClient.expect(b'done')
    self._testwrite(b"class Foo:\n\tdef bar(self):\n\t\tprint('Hello, world!')\n\nFoo().bar()\ndone")

    def finished(ign):
        self._assertBuffer([b'>>> class Foo:', b'...     def bar(self):', b"...         print('Hello, world!')", b'... ', b'>>> Foo().bar()', b'Hello, world!', b'>>> done'])
    return done.addCallback(finished)