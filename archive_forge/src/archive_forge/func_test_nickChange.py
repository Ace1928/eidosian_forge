import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_nickChange(self):
    """
        When a NICK command is sent after signon, C{IRCClient.nickname} is set
        to the new nickname I{after} the server sends an acknowledgement.
        """
    oldnick = 'foo'
    newnick = 'bar'
    self.protocol.register(oldnick)
    self.protocol.irc_RPL_WELCOME('prefix', ['param'])
    self.protocol.setNick(newnick)
    self.assertEqual(self.protocol.nickname, oldnick)
    self.protocol.irc_NICK(f'{oldnick}!quux@qux', [newnick])
    self.assertEqual(self.protocol.nickname, newnick)