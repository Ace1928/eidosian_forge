import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_registerWithTakenNick(self):
    """
        Verify that the client repeats the L{IRCClient.setNick} method with a
        new value when presented with an C{ERR_NICKNAMEINUSE} while trying to
        register.
        """
    username = 'testuser'
    hostname = 'testhost'
    servername = 'testserver'
    self.protocol.realname = 'testname'
    self.protocol.password = 'testpass'
    self.protocol.register(username, hostname, servername)
    self.protocol.irc_ERR_NICKNAMEINUSE('prefix', ['param'])
    lastLine = self.getLastLine(self.transport)
    self.assertNotEqual(lastLine, f'NICK {username}')
    self.protocol.irc_ERR_NICKNAMEINUSE('prefix', ['param'])
    lastLine = self.getLastLine(self.transport)
    self.assertEqual(lastLine, 'NICK {}'.format(username + '__'))