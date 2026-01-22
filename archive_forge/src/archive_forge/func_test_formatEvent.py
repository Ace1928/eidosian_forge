from typing import AnyStr, Dict, Optional, cast
from twisted.python.failure import Failure
from twisted.python.test.test_tzhelper import addTZCleanup, mktime, setTZ
from twisted.trial import unittest
from twisted.trial.unittest import SkipTest
from .._format import (
from .._interfaces import LogEvent
from .._levels import LogLevel
def test_formatEvent(self) -> None:
    """
        L{formatEvent} will format an event according to several rules:

            - A string with no formatting instructions will be passed straight
              through.

            - PEP 3101 strings will be formatted using the keys and values of
              the event as named fields.

            - PEP 3101 keys ending with C{()} will be treated as instructions
              to call that key (which ought to be a callable) before
              formatting.

        L{formatEvent} will always return L{str}, and if given bytes, will
        always treat its format string as UTF-8 encoded.
        """
    self.assertEqual('', self.format(b''))
    self.assertEqual('', self.format(''))
    self.assertEqual('abc', self.format('{x}', x='abc'))
    self.assertEqual('no, yes.', self.format('{not_called}, {called()}.', not_called='no', called=lambda: 'yes'))
    self.assertEqual('Sánchez', self.format(b'S\xc3\xa1nchez'))
    self.assertIn('Unable to format event', self.format(b'S\xe1nchez'))
    maybeResult = self.format(b'S{a!s}nchez', a=b'\xe1')
    self.assertIn("Sb'\\xe1'nchez", maybeResult)
    xe1 = str(repr(b'\xe1'))
    self.assertIn('S' + xe1 + 'nchez', self.format(b'S{a!r}nchez', a=b'\xe1'))