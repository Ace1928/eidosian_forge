import errno
import os
import re
import sys
from inspect import getmro
from io import BytesIO, StringIO
from typing import Type
from unittest import (
from hamcrest import assert_that, equal_to, has_item, has_length
from twisted.python import log
from twisted.python.failure import Failure
from twisted.trial import itrial, reporter, runner, unittest, util
from twisted.trial.reporter import UncleanWarningsReporterWrapper, _ExitWrapper
from twisted.trial.test import erroneous, sample
from twisted.trial.unittest import SkipTest, Todo, makeTodo
from .._dist.test.matchers import isFailure, matches_result, similarFrame
from .matchers import after
def test_addExpectedFailurePassedThrough(self):
    """
        Some versions of subunit have "addExpectedFailure". For these
        versions, when we call 'addExpectedFailure' on the test result, we
        pass the error and test through to the subunit client.
        """
    addExpectedFailureCalls = []

    def addExpectedFailure(test, error):
        addExpectedFailureCalls.append((test, error))
    self.result._subunit.addExpectedFailure = addExpectedFailure
    try:
        1 / 0
    except ZeroDivisionError:
        exc_info = sys.exc_info()
        self.result.addExpectedFailure(self.test, exc_info, 'todo')
    self.assertEqual(addExpectedFailureCalls, [(self.test, exc_info)])