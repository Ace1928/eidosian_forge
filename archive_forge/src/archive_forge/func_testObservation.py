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
def testObservation(self) -> None:
    catcher = self.catcher
    log.msg('test', testShouldCatch=True)
    i = catcher.pop()
    self.assertEqual(i['message'][0], 'test')
    self.assertTrue(i['testShouldCatch'])
    self.assertIn('time', i)
    self.assertEqual(len(catcher), 0)