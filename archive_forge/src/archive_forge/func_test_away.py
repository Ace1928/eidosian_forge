import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_away(self):
    """
        L{IRCClient.away} sends an AWAY command with the specified message.
        """
    message = "Sorry, I'm not here."
    self.protocol.away(message)
    expected = [f'AWAY :{message}', '']
    self.assertEqualBufferValue(self.transport.value().split(b'\r\n'), expected)