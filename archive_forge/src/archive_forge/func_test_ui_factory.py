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
def test_ui_factory(self):
    current_factory = breezy.ui.ui_factory
    self.run_bzr(['foo'])
    self.assertFalse(current_factory is self.factory)
    self.assertNotEqual(sys.stdout, self.factory.stdout)
    self.assertNotEqual(sys.stderr, self.factory.stderr)
    self.assertEqual('foo\n', self.factory.stdout.getvalue())
    self.assertEqual('bar\n', self.factory.stderr.getvalue())
    self.assertIsInstance(self.factory, tests.TestUIFactory)