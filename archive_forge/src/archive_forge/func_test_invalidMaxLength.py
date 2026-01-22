import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_invalidMaxLength(self):
    """
        Specifying a C{length} value to L{IRCClient.msg} that is too short to
        contain the protocol command to send a message raises C{ValueError}.
        """
    self.assertRaises(ValueError, self.client.msg, 'foo', 'bar', 0)
    self.assertRaises(ValueError, self.client.msg, 'foo', 'bar', 3)