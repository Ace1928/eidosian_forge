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
def test__gather_lsprof_in_benchmarks(self):
    """When _gather_lsprof_in_benchmarks is on, accumulate profile data.

        Each self.time() call is individually and separately profiled.
        """
    self.requireFeature(features.lsprof_feature)
    self._gather_lsprof_in_benchmarks = True
    self.time(time.sleep, 0.0)
    self.time(time.sleep, 0.003)
    self.assertEqual(2, len(self._benchcalls))
    self.assertEqual((time.sleep, (0.0,), {}), self._benchcalls[0][0])
    self.assertEqual((time.sleep, (0.003,), {}), self._benchcalls[1][0])
    self.assertIsInstance(self._benchcalls[0][1], breezy.lsprof.Stats)
    self.assertIsInstance(self._benchcalls[1][1], breezy.lsprof.Stats)
    del self._benchcalls[:]