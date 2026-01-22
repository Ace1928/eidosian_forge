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
def test_lsprofiling(self):
    """Verbose test result prints lsprof statistics from test cases."""
    self.requireFeature(features.lsprof_feature)
    result_stream = StringIO()
    result = breezy.tests.VerboseTestResult(result_stream, descriptions=0, verbosity=2)
    example_test_case = TestTestResult('_time_hello_world_encoding')
    example_test_case._gather_lsprof_in_benchmarks = True
    example_test_case.run(result)
    output = result_stream.getvalue()
    self.assertContainsRe(output, "LSProf output for <class 'str'>\\(\\(b'hello',\\), {'errors': 'replace'}\\)")
    self.assertContainsRe(output, "LSProf output for <class 'str'>\\(\\(b'world',\\), {'errors': 'replace'}\\)")
    self.assertContainsRe(output, ' *CallCount *Recursive *Total\\(ms\\) *Inline\\(ms\\) *module:lineno\\(function\\)\\n')
    self.assertContainsRe(output, "( +1 +0 +0\\.\\d+ +0\\.\\d+ +<method 'disable' of '_lsprof\\.Profiler' objects>\\n)?")