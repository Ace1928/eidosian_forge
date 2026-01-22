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
def test_startStopObserver(self) -> None:
    """
        Test that start and stop methods of the observer actually register
        and unregister to the log system.
        """
    oldAddObserver = log.addObserver
    oldRemoveObserver = log.removeObserver
    l: list[Callable[[log.EventDict], None]] = []
    try:
        log.addObserver = l.append
        log.removeObserver = l.remove
        obs = log.PythonLoggingObserver()
        obs.start()
        self.assertEqual(l[0], obs.emit)
        obs.stop()
        self.assertEqual(len(l), 0)
    finally:
        log.addObserver = oldAddObserver
        log.removeObserver = oldRemoveObserver