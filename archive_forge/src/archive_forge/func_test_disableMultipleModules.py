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
def test_disableMultipleModules(self) -> None:
    """
        Check that several modules can be disabled at once.
        """
    self.config.parseOptions(['--without-module', 'smtplib,imaplib'])
    self.assertRaises(ImportError, self._checkSMTP)
    self.assertRaises(ImportError, self._checkIMAP)
    del sys.modules['smtplib']
    del sys.modules['imaplib']
    self.assertIsInstance(self._checkSMTP(), types.ModuleType)
    self.assertIsInstance(self._checkIMAP(), types.ModuleType)