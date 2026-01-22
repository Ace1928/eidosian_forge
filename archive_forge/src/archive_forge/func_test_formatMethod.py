from typing import AnyStr, Dict, Optional, cast
from twisted.python.failure import Failure
from twisted.python.test.test_tzhelper import addTZCleanup, mktime, setTZ
from twisted.trial import unittest
from twisted.trial.unittest import SkipTest
from .._format import (
from .._interfaces import LogEvent
from .._levels import LogLevel
def test_formatMethod(self) -> None:
    """
        L{formatEvent} will format PEP 3101 keys containing C{.}s ending with
        C{()} as methods.
        """

    class World:

        def where(self) -> str:
            return 'world'
    self.assertEqual('hello world', self.format('hello {what.where()}', what=World()))