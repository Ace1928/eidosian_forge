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
def test_test_suite_matches_id_list_with_unknown(self):
    loader = TestUtil.TestLoader()
    suite = loader.loadTestsFromModuleName('breezy.tests.test_sampler')
    test_list = ['breezy.tests.test_sampler.DemoTest.test_nothing', 'bogus']
    not_found, duplicates = tests.suite_matches_id_list(suite, test_list)
    self.assertEqual(['bogus'], not_found)
    self.assertEqual([], duplicates)