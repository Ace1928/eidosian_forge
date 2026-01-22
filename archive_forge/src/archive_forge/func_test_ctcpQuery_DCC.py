import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_ctcpQuery_DCC(self):
    """
        L{IRCClient.ctcpQuery_DCC} calls L{IRCClient.ctcpMakeReply} with the
        correct args.
        """
    self.client.ctcpQuery_DCC(self.user, self.channel, 'data')
    self.assertEqual(self.client.methods, [('ctcpMakeReply', ('Wolf', [('ERRMSG', "DCC data :Unknown DCC type 'DATA'")]))])