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
def test_help_ordersPrintsSynopsisAndQuits(self) -> None:
    """
        --help-orders prints each of the available orders and then exits.
        """
    self.patch(sys, 'stdout', (stdout := StringIO()))
    exc = self.assertRaises(SystemExit, trial.Options().parseOptions, ['--help-orders'])
    self.assertEqual(exc.code, 0)
    output = stdout.getvalue()
    msg = '%r with its description not properly described in %r'
    for orderName, (orderDesc, _) in trial._runOrders.items():
        match = re.search(f'{re.escape(orderName)}.*{re.escape(orderDesc)}', output)
        self.assertTrue(match, msg=msg % (orderName, output))