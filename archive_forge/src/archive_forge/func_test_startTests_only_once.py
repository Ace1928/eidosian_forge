import doctest
import gc
import os
import signal
import sys
import threading
import time
import unittest
import warnings
from functools import reduce
from io import BytesIO, StringIO, TextIOWrapper
import testtools.testresult.doubles
from testtools import ExtendedToOriginalDecorator, MultiTestResult
from testtools.content import Content
from testtools.content_type import ContentType
from testtools.matchers import DocTestMatches, Equals
import breezy
from .. import (branchbuilder, controldir, errors, hooks, lockdir, memorytree,
from ..bzr import (bzrdir, groupcompress_repo, remote, workingtree_3,
from ..git import workingtree as git_workingtree
from ..symbol_versioning import (deprecated_function, deprecated_in,
from ..trace import mutter, note
from ..transport import memory
from . import TestUtil, features, test_lsprof, test_server
def test_startTests_only_once(self):
    """With multiple tests startTests should still only be called once"""

    class InstrumentedTestResult(tests.ExtendedTestResult):
        calls = 0

        def startTests(self):
            self.calls += 1
    result = InstrumentedTestResult(None, None, None, None)
    suite = unittest.TestSuite([unittest.FunctionTestCase(lambda: None), unittest.FunctionTestCase(lambda: None)])
    suite.run(result)
    self.assertEqual(1, result.calls)
    self.assertEqual(2, result.count)