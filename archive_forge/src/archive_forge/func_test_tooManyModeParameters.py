import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_tooManyModeParameters(self):
    """
        Passing an argument to modes that take no parameters results in
        L{IRCClient.modeChanged} not being called and an error being logged.
        """
    self._sendModeChange('+s', 'wrong')
    self._checkModeChange([])
    errors = self.flushLoggedErrors(irc.IRCBadModes)
    self.assertEqual(len(errors), 1)
    self.assertSubstring('Too many parameters', errors[0].getErrorMessage())