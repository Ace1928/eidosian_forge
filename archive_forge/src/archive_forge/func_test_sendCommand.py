import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_sendCommand(self):
    """
        Passing a command and parameters to L{IRC.sendCommand} results in a
        query string that consists of the command and parameters, separated by
        a space, ending with '\r
'.

        The format is described in more detail in
        U{RFC 1459 <https://tools.ietf.org/html/rfc1459.html#section-2.3>}.
        """
    self.p.sendCommand('CMD', ('param1', 'param2'))
    self.check('CMD param1 param2\r\n')