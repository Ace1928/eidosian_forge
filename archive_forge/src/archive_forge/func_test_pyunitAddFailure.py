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
def test_pyunitAddFailure(self):
    try:
        raise self.failureException('foo')
    except self.failureException as e:
        excValue = e
        self.result.addFailure(self, sys.exc_info())
    failure = self.result.failures[0][1]
    self.assertEqual(excValue, failure.value)
    self.assertEqual(self.failureException, failure.type)