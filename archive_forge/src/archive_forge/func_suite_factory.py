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
def suite_factory(self, keep_only=None, starting_with=None):
    """A test suite factory with only a few tests."""

    class Test(tests.TestCase):

        def id(self):
            return self._testMethodName

        def a(self):
            pass

        def b(self):
            pass

        def c(self):
            pass
    return TestUtil.TestSuite([Test('a'), Test('b'), Test('c')])