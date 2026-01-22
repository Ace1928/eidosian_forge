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
def test_randomize_suite(self):
    randomized_suite = tests.randomize_suite(self.suite)
    self.assertEqual(set(_test_ids(self.suite)), set(_test_ids(randomized_suite)))
    self.assertNotEqual(self.all_names, _test_ids(randomized_suite))
    self.assertEqual(len(self.all_names), len(_test_ids(randomized_suite)))