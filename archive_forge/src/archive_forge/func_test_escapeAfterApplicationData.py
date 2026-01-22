import textwrap
from typing import Optional, Type
from twisted.conch.insults.insults import (
from twisted.internet.protocol import Protocol
from twisted.internet.testing import StringTransport
from twisted.python.compat import iterbytes
from twisted.python.constants import ValueConstant, Values
from twisted.trial import unittest
def test_escapeAfterApplicationData(self):
    """
        Application data bytes followed by an escape character are passed to a
        call to C{write} before the terminal's handler method for the escape is
        called.
        """
    self._applicationDataTest(b'ab\x1bD', [('write', (b'ab',)), ('index',)])
    self._applicationDataTest(b'ab\x1b[4h', [('write', (b'ab',)), ('setModes', ([4],))])