import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_modeMissingDirection(self):
    """
        Mode strings that do not begin with a directional character, C{'+'} or
        C{'-'}, have C{'+'} automatically prepended.
        """
    self._sendModeChange('s')
    self._checkModeChange([(True, 's', (None,))])