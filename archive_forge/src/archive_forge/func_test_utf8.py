from __future__ import annotations
import locale
import os
import sys
from io import StringIO
from typing import Generator
from zope.interface import implementer
from hamcrest import assert_that, equal_to
from twisted.internet.base import DelayedCall
from twisted.internet.interfaces import IProcessTransport
from twisted.python import filepath
from twisted.python.failure import Failure
from twisted.trial import util
from twisted.trial.unittest import SynchronousTestCase
from twisted.trial.util import (
def test_utf8(self) -> None:
    """
        The log file is opened in text mode and uses UTF-8 for encoding.
        """
    currentLocale = locale.getlocale()
    self.addCleanup(locale.setlocale, locale.LC_ALL, currentLocale)
    locale.setlocale(locale.LC_ALL, ('C', 'ascii'))
    text = 'Here comes the ☉'
    p = filepath.FilePath(self.mktemp())
    with openTestLog(p) as f:
        f.write(text)
    with open(p.path, 'rb') as f:
        written = f.read()
    assert_that(text.encode('utf-8'), equal_to(written))