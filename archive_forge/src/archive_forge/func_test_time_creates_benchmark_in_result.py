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
def test_time_creates_benchmark_in_result(self):
    """The TestCase.time() method accumulates a benchmark time."""
    sample_test = TestTestCase('method_that_times_a_bit_twice')
    output_stream = StringIO()
    result = breezy.tests.VerboseTestResult(output_stream, descriptions=0, verbosity=2)
    sample_test.run(result)
    self.assertContainsRe(output_stream.getvalue(), '\\d+ms\\*\\n$')