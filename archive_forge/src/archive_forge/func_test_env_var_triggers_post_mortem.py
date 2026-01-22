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
def test_env_var_triggers_post_mortem(self):
    """Check pdb.post_mortem is called iff BRZ_TEST_PDB is set"""
    import pdb
    result = tests.ExtendedTestResult(StringIO(), 0, 1)
    post_mortem_calls = []
    self.overrideAttr(pdb, 'post_mortem', post_mortem_calls.append)
    self.overrideEnv('BRZ_TEST_PDB', None)
    result._post_mortem(1)
    self.overrideEnv('BRZ_TEST_PDB', 'on')
    result._post_mortem(2)
    self.assertEqual([2], post_mortem_calls)