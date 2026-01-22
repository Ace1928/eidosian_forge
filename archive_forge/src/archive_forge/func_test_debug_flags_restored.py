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
def test_debug_flags_restored(self):
    """The breezy debug flags should be restored to their original state
        after the test was run, even if allow_debug is set.
        """
    self.change_selftest_debug_flags({'allow_debug'})
    breezy.debug.debug_flags = {'original-state'}

    class TestThatModifiesFlags(tests.TestCase):

        def test_foo(self):
            breezy.debug.debug_flags = {'modified'}
    test = TestThatModifiesFlags('test_foo')
    test.run(self.make_test_result())
    self.assertEqual({'original-state'}, breezy.debug.debug_flags)