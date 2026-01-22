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
def test_emitPrefix(self) -> None:
    """
        FileLogObserver.emit() will add a timestamp and system prefix to its
        file output.
        """
    output = StringIO()
    flo = log.FileLogObserver(output)
    events = []

    def observer(event: log.EventDict) -> None:
        events.append(event)
        flo.emit(event)
    publisher = log.LogPublisher()
    publisher.addObserver(observer)
    publisher.msg('Hello!')
    self.assertEqual(len(events), 1)
    event = events[0]
    result = output.getvalue()
    prefix = '{time} [{system}] '.format(time=flo.formatTime(event['time']), system=event['system'])
    self.assertTrue(result.startswith(prefix), f'{result!r} does not start with {prefix!r}')