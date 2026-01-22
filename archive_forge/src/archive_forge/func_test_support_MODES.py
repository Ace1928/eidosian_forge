import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_support_MODES(self):
    """
        The MODES support parameter is parsed into an integer value
        indicating the maximum number of "variable" modes (defined as being
        modes from C{addressModes}, C{param} or C{setParam} categories for
        the C{CHANMODES} ISUPPORT parameter) which may by set on a channel
        by a single MODE command from a client.
        """
    self._testIntOrDefaultFeature('MODES')