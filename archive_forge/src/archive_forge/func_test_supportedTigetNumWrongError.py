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
def test_supportedTigetNumWrongError(self):
    """
        L{reporter._AnsiColorizer.supported} returns C{False} and doesn't try
        to call C{curses.setupterm} if C{curses.tigetnum} returns something
        different than C{curses.error}.
        """

    class fakecurses:
        error = RuntimeError

        def tigetnum(self, value):
            raise ValueError()
    sys.modules['curses'] = fakecurses()
    self.assertFalse(reporter._AnsiColorizer.supported(FakeStream()))