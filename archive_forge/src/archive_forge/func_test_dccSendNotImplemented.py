import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_dccSendNotImplemented(self):
    """
        L{irc.IRCClient.dccDoSend} is raises C{NotImplementedError}
        """
    client = irc.IRCClient()
    self.assertRaises(NotImplementedError, client.dccSend, 'username', None)