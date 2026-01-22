import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_tooFewModeParameters(self):
    """
        Passing no arguments to modes that do take parameters results in
        L{IRCClient.modeChange} not being called and an error being logged.
        """
    self._sendModeChange('+o')
    self._checkModeChange([])
    errors = self.flushLoggedErrors(irc.IRCBadModes)
    self.assertEqual(len(errors), 1)
    self.assertSubstring('Not enough parameters', errors[0].getErrorMessage())