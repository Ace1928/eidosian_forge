from __future__ import annotations
import gc
import re
import sys
import textwrap
import types
from io import StringIO
from typing import List
from hamcrest import assert_that, contains_string
from hypothesis import given
from hypothesis.strategies import sampled_from
from twisted.logger import Logger
from twisted.python import util
from twisted.python.filepath import FilePath, IFilePath
from twisted.python.usage import UsageError
from twisted.scripts import trial
from twisted.trial import unittest
from twisted.trial._dist.disttrial import DistTrialRunner
from twisted.trial._dist.functional import compose
from twisted.trial.runner import (
from twisted.trial.test.test_loader import testNames
from .matchers import fileContents
@given(sampled_from(['somelog.txt', 'somedir/somelog.txt']))
def test_absolutePath(self, logfile: str) -> None:
    """
        If the value given for the option is an absolute path then it is
        interpreted absolutely and logs are written there.
        """
    logPath = FilePath('.').preauthChild(logfile)
    runFromArguments(['--logfile', logPath.path])
    iPath: IFilePath = logPath
    assert_that(iPath, fileContents(contains_string('something')))