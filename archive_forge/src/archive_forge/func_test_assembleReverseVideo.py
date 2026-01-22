import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_assembleReverseVideo(self):
    """
        A I{reverse video} string assembles to a string prefixed with the I{off}
        and I{reverse video} control codes.
        """
    self.assertEqual(irc.assembleFormattedText(A.reverseVideo['hello']), '\x0f\x16hello')