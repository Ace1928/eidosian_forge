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
def test_stringTypeError(self) -> None:
    """
        If things is a string, a TypeError is raised.
        """
    sample = 'One, two, three'
    error = self.assertRaises(TypeError, util._listToPhrase, sample, 'and')
    self.assertEqual(str(error), 'Things must be a list or a tuple')