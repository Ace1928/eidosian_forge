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
def test_home_is_non_existant_dir_under_root(self):
    """The test_home_dir for TestCaseWithMemoryTransport is missing.

        This is because TestCaseWithMemoryTransport is for tests that do not
        need any disk resources: they should be hooked into breezy in such a
        way that no global settings are being changed by the test (only a
        few tests should need to do that), and having a missing dir as home is
        an effective way to ensure that this is the case.
        """
    self.assertIsSameRealPath(self.TEST_ROOT + '/MemoryTransportMissingHomeDir', self.test_home_dir)
    self.assertIsSameRealPath(self.test_home_dir, os.environ['HOME'])