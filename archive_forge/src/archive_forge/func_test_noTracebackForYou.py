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
def test_noTracebackForYou(self) -> None:
    """
        If unable to obtain a traceback due to an exception, catch it and note
        the error.
        """
    eventDict = dict(message=(), isError=1, failure=object())
    text = log.textFromEventDict(eventDict)
    self.assertIn('\n(unable to obtain traceback)', text)