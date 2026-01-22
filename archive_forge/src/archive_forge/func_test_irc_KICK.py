import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_irc_KICK(self):
    """
        L{IRCClient.kickedFrom} is called when I get kicked from the channel;
        L{IRCClient.userKicked} is called when someone else gets kicked.
        """
    self.client.irc_KICK('Svadilfari!~svadi@yok.utu.fi', ['#python', 'WOLF', 'shoryuken!'])
    self.client.irc_KICK(self.user, [self.channel, 'Svadilfari', 'hadouken!'])
    self.assertEqual(self.client.methods, [('kickedFrom', ('#python', 'Svadilfari', 'shoryuken!')), ('userKicked', ('Svadilfari', self.channel, 'Wolf', 'hadouken!'))])