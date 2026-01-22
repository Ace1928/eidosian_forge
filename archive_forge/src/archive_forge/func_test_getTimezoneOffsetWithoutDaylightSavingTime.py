from __future__ import annotations
import calendar
import logging
import os
import sys
import time
import warnings
from io import IOBase, StringIO
from typing import Callable, List
from zope.interface import implementer
from typing_extensions import Protocol
from twisted.logger import (
from twisted.logger.test.test_stdlib import handlerAndBytesIO
from twisted.python import failure, log
from twisted.python.log import LogPublisher
from twisted.trial import unittest
def test_getTimezoneOffsetWithoutDaylightSavingTime(self) -> None:
    """
        Attempt to verify that L{FileLogObserver.getTimezoneOffset} returns
        correct values for the current C{TZ} environment setting for at least
        some cases.  This test method exercises a timezone that does not use
        daylight saving time at all (so both summer and winter time test values
        should have the same offset).
        """
    self._getTimezoneOffsetTest('Africa/Johannesburg', -7200, -7200)