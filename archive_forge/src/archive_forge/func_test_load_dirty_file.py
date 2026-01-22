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
def test_load_dirty_file(self):
    test_list_fname = 'test.list'
    self._create_test_list_file(test_list_fname, '  mod1.cl1.meth1\n\nmod2.cl2.meth2  \nbar baz\n')
    tlist = tests.load_test_id_list(test_list_fname)
    self.assertEqual(4, len(tlist))
    self.assertEqual('mod1.cl1.meth1', tlist[0])
    self.assertEqual('', tlist[1])
    self.assertEqual('mod2.cl2.meth2', tlist[2])
    self.assertEqual('bar baz', tlist[3])