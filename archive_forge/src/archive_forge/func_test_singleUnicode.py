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
def test_singleUnicode(self) -> None:
    """
        L{log.LogPublisher.msg} does not accept non-ASCII Unicode on Python 2,
        logging an error instead.

        On Python 3, where Unicode is default message type, the message is
        logged normally.
        """
    message = 'Hello, ½ world.'
    self.lp.msg(message)
    self.assertEqual(len(self.out), 1)
    self.assertIn(message, self.out[0])