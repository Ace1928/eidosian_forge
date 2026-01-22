import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_whoisWithServer(self):
    """
        L{IRCClient.whois} sends a WHOIS message with a server name if a
        value is passed for the C{server} parameter.
        """
    self.protocol.whois('alice', 'example.org')
    self.assertEqualBufferValue(self.transport.value().split(b'\r\n'), ['WHOIS example.org alice', ''])