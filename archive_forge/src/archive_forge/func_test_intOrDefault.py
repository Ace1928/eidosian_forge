import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_intOrDefault(self):
    """
        L{_intOrDefault} converts values to C{int} if possible, otherwise
        returns a default value.
        """
    self.assertEqual(irc._intOrDefault(None), None)
    self.assertEqual(irc._intOrDefault([]), None)
    self.assertEqual(irc._intOrDefault(''), None)
    self.assertEqual(irc._intOrDefault('hello', 5), 5)
    self.assertEqual(irc._intOrDefault('123'), 123)
    self.assertEqual(irc._intOrDefault(123), 123)