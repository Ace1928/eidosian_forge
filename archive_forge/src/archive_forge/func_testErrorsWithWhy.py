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
def testErrorsWithWhy(self) -> None:
    for e, ig in [('hello world', 'hello world'), (KeyError(), KeyError), (failure.Failure(RuntimeError()), RuntimeError)]:
        log.err(e, 'foobar')
        i = self.catcher.pop()
        self.assertEqual(i['isError'], 1)
        self.assertEqual(i['why'], 'foobar')
        self.flushLoggedErrors(ig)