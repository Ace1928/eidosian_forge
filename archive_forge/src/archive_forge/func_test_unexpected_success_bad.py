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
def test_unexpected_success_bad(self):

    class Test(tests.TestCase):

        def test_truth(self):
            self.expectFailure('No absolute truth', self.assertTrue, True)
    runner = tests.TextTestRunner(stream=StringIO())
    self.run_test_runner(runner, Test('test_truth'))
    self.assertContainsRe(runner.stream.getvalue(), '=+\nFAIL: \\S+\\.test_truth\n-+\n(?:.*\n)*\\s*(?:Text attachment: )?reason(?:\n-+\n|: {{{)No absolute truth(?:\n-+\n|}}}\n)(?:.*\n)*-+\nRan 1 test in .*\n\nFAILED \\(failures=1\\)\n\\Z')