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
def test_not_applicable(self):

    class Test(tests.TestCase):

        def not_applicable_test(self):
            raise tests.TestNotApplicable('this test never runs')
    out = StringIO()
    runner = tests.TextTestRunner(stream=out, verbosity=2)
    test = Test('not_applicable_test')
    result = self.run_test_runner(runner, test)
    self.log(out.getvalue())
    self.assertTrue(result.wasSuccessful())
    self.assertTrue(result.wasStrictlySuccessful())
    self.assertContainsRe(out.getvalue(), '(?m)not_applicable_test  * N/A')
    self.assertContainsRe(out.getvalue(), '(?m)^    this test never runs')