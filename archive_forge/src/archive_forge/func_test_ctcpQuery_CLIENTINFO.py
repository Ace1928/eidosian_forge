import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_ctcpQuery_CLIENTINFO(self):
    """
        L{IRCClient.ctcpQuery_CLIENTINFO} calls L{IRCClient.ctcpMakeReply} with
        the correct args.
        """
    self.client.ctcpQuery_CLIENTINFO(self.user, self.channel, '')
    self.client.ctcpQuery_CLIENTINFO(self.user, self.channel, 'PING PONG')
    info = 'ACTION CLIENTINFO DCC ERRMSG FINGER PING SOURCE TIME USERINFO VERSION'
    self.assertEqual(self.client.methods, [('ctcpMakeReply', ('Wolf', [('CLIENTINFO', info)])), ('ctcpMakeReply', ('Wolf', [('CLIENTINFO', None)]))])