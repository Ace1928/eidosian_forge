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
def test_unexpectedSuccess(self):
    """
        A test which is marked as todo but succeeds will have an unexpected
        success reported to its result. A test run is still successful even
        when this happens.
        """
    self.result.addUnexpectedSuccess(self.test, makeTodo('Heya!'))
    self.assertEqual(True, self.result.wasSuccessful())
    self.assertEqual(len(self._getUnexpectedSuccesses(self.result)), 1)