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
def test_minimalReporter(self):
    """
        The summary of L{reporter.MinimalReporter} is a simple list of numbers,
        indicating how many tests ran, how many failed etc.

        The numbers represents:
         * the run time of the tests
         * the number of tests run, printed 2 times for legacy reasons
         * the number of errors
         * the number of failures
         * the number of skips
        """
    result = reporter.MinimalReporter(self.stream)
    self.test.run(result)
    result._printSummary()
    output = self.stream.getvalue().strip().split(' ')
    self.assertEqual(output[1:], ['1', '1', '0', '0', '0'])