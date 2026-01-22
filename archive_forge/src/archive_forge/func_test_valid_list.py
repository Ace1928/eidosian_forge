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
def test_valid_list(self):
    id_list = self._create_id_list(['mod1.cl1.meth1', 'mod1.cl1.meth2', 'mod1.func1', 'mod1.cl2.meth2', 'mod1.submod1', 'mod1.submod2.cl1.meth1', 'mod1.submod2.cl2.meth2'])
    self.assertTrue(id_list.refers_to('mod1'))
    self.assertTrue(id_list.refers_to('mod1.submod1'))
    self.assertTrue(id_list.refers_to('mod1.submod2'))
    self.assertTrue(id_list.includes('mod1.cl1.meth1'))
    self.assertTrue(id_list.includes('mod1.submod1'))
    self.assertTrue(id_list.includes('mod1.func1'))