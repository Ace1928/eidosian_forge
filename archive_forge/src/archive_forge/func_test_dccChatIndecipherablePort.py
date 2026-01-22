import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_dccChatIndecipherablePort(self):
    """
        L{irc.IRCClient.dcc_CHAT} raises L{irc.IRCBadMessage} when it is passed
        a query string that doesn't contain a valid port number.
        """
    result = self.assertRaises(irc.IRCBadMessage, self.client.dcc_CHAT, self.user, self.channel, 'foo.txt 127.0.0.1 sd@d')
    self.assertEqual(str(result), "Indecipherable port 'sd@d'")