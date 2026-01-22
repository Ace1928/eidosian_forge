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
def test_assert_isinstance(self):
    self.assertIsInstance(2, int)
    self.assertIsInstance('', str)
    e = self.assertRaises(AssertionError, self.assertIsInstance, None, int)
    self.assertIn(str(e), ["None is an instance of <type 'NoneType'> rather than <type 'int'>", "None is an instance of <class 'NoneType'> rather than <class 'int'>"])
    self.assertRaises(AssertionError, self.assertIsInstance, 23.3, int)
    e = self.assertRaises(AssertionError, self.assertIsInstance, None, int, "it's just not")
    self.assertEqual(str(e), "None is an instance of <class 'NoneType'> rather than <class 'int'>: it's just not")