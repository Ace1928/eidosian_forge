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
def test_whySpecified(self) -> None:
    """
        The C{"why"} value, when specified, is first part of message.
        """
    try:
        raise RuntimeError()
    except BaseException:
        eventDict = dict(message=(), isError=1, failure=failure.Failure(), why='foo')
        text = log.textFromEventDict(eventDict)
        assert text is not None
        self.assertTrue(text.startswith('foo\n'))