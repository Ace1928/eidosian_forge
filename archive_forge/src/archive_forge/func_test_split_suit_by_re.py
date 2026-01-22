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
def test_split_suit_by_re(self):
    self.all_names = _test_ids(self.suite)
    split_suite = tests.split_suite_by_re(self.suite, 'test_filter_suite_by_r')
    filtered_name = 'breezy.tests.test_selftest.TestSelftestFiltering.test_filter_suite_by_re'
    self.assertEqual([filtered_name], _test_ids(split_suite[0]))
    self.assertFalse(filtered_name in _test_ids(split_suite[1]))
    remaining_names = list(self.all_names)
    remaining_names.remove(filtered_name)
    self.assertEqual(remaining_names, _test_ids(split_suite[1]))