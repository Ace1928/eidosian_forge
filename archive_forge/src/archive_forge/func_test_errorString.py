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
def test_errorString(self) -> None:
    """
        Test error output.
        """
    f = failure.Failure(ValueError('That is bad.'))
    self.lp.msg(failure=f, isError=True)
    prefix = b'CRITICAL:'
    output = self.out.getvalue()
    self.assertTrue(output.startswith(prefix), f'Does not start with {prefix!r}: {output!r}')