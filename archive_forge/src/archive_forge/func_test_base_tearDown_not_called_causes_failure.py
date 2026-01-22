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
def test_base_tearDown_not_called_causes_failure(self):

    class TestCaseWithBrokenTearDown(tests.TestCase):

        def tearDown(self):
            pass

        def test_foo(self):
            pass
    test = TestCaseWithBrokenTearDown('test_foo')
    result = unittest.TestResult()
    test.run(result)
    self.assertFalse(result.wasSuccessful())
    self.assertEqual(1, result.testsRun)