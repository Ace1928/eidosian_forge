import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_dccSendMalformedRequest(self):
    """
        L{irc.IRCClient.dcc_SEND} raises L{irc.IRCBadMessage} when it is passed
        a malformed query string.
        """
    result = self.assertRaises(irc.IRCBadMessage, self.client.dcc_SEND, self.user, self.channel, 'foo')
    self.assertEqual(str(result), "malformed DCC SEND request: ['foo']")