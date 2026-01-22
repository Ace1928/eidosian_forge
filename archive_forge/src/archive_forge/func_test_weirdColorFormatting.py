import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_weirdColorFormatting(self):
    """
        Formatted text with colors can use 1 digit for both foreground and
        background, as long as the text part does not begin with a digit.
        Foreground and background colors are only processed to a maximum of 2
        digits per component, anything else is treated as text. Color sequences
        must begin with a digit, otherwise processing falls back to unformatted
        text.
        """
    self.assertAssembledEqually('\x031kinda valid', A.fg.black['kinda valid'])
    self.assertAssembledEqually('\x03999,999kinda valid', A.fg.green['9,999kinda valid'])
    self.assertAssembledEqually('\x031,2kinda valid', A.fg.black[A.bg.blue['kinda valid']])
    self.assertAssembledEqually('\x031,999kinda valid', A.fg.black[A.bg.green['9kinda valid']])
    self.assertAssembledEqually('\x031,242 is a special number', A.fg.black[A.bg.yellow['2 is a special number']])
    self.assertAssembledEqually('\x03,02oops\x03', A.normal[',02oops'])
    self.assertAssembledEqually('\x03wrong', A.normal['wrong'])
    self.assertAssembledEqually('\x031,hello', A.fg.black['hello'])
    self.assertAssembledEqually('\x03\x03', A.normal)