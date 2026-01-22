import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_sendCommandWithPrefix(self):
    """
        Passing a command and parameters with a specified prefix to
        L{IRC.sendCommand} results in a proper query string including the
        specified line prefix.
        """
    self.p.sendCommand('CMD', ('param1', 'param2'), 'irc.example.com')
    self.check(b':irc.example.com CMD param1 param2\r\n')