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
def test_groupResults(self):
    """
        If two different tests have the same error, L{Reporter._groupResults}
        includes them together in one of the tuples in the list it returns.
        """
    try:
        raise RuntimeError('foo')
    except RuntimeError:
        self.result.addError(self, sys.exc_info())
        self.result.addError(self.test, sys.exc_info())
    try:
        raise RuntimeError('bar')
    except RuntimeError:
        extra = sample.FooTest('test_bar')
        self.result.addError(extra, sys.exc_info())
    self.result.done()
    grouped = self.result._groupResults(self.result.errors, self.result._formatFailureTraceback)
    self.assertEqual(grouped[0][1], [self, self.test])
    self.assertEqual(grouped[1][1], [extra])