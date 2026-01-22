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
def test_showwarning(self) -> None:
    """
        L{twisted.python.log.showwarning} emits the warning as a message
        to the Twisted logging system.
        """
    publisher = log.LogPublisher()
    publisher.addObserver(self.observer)
    publisher.showwarning(FakeWarning('unique warning message'), FakeWarning, 'warning-filename.py', 27)
    event = self.catcher.pop()
    self.assertEqual(event['format'] % event, 'warning-filename.py:27: twisted.test.test_log.FakeWarning: unique warning message')
    self.assertEqual(self.catcher, [])
    publisher.showwarning(FakeWarning('unique warning message'), FakeWarning, 'warning-filename.py', 27, line=object())
    event = self.catcher.pop()
    self.assertEqual(event['format'] % event, 'warning-filename.py:27: twisted.test.test_log.FakeWarning: unique warning message')
    self.assertEqual(self.catcher, [])