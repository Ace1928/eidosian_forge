import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_receivedMOTD(self):
    """
        Lines received in I{RPL_MOTDSTART} and I{RPL_MOTD} are delivered to
        L{IRCClient.receivedMOTD} when I{RPL_ENDOFMOTD} is received.
        """
    lines = [':host.name 375 nickname :- host.name Message of the Day -', ':host.name 372 nickname :- Welcome to host.name', ':host.name 376 nickname :End of /MOTD command.']
    for L in lines:
        self.assertEqual(self.client.calls, [])
        self.client.dataReceived(L + '\r\n')
    self.assertEqual(self.client.calls, [('receivedMOTD', {'motd': ['host.name Message of the Day -', 'Welcome to host.name']})])
    self.assertIdentical(self.client.motd, None)