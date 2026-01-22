import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_irc_QUIT(self):
    """
        L{IRCClient.userQuit} is called whenever someone quits
        the channel (myself included).
        """
    self.client.irc_QUIT('Svadilfari!~svadi@yok.utu.fi', ['Adios.'])
    self.client.irc_QUIT(self.user, ['Farewell.'])
    self.assertEqual(self.client.methods, [('userQuit', ('Svadilfari', 'Adios.')), ('userQuit', ('Wolf', 'Farewell.'))])