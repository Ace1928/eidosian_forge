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
def test_overrideAttr_with_no_existing_value_and_no_value(self):
    obj = self

    class Test(tests.TestCase):

        def setUp(self):
            tests.TestCase.setUp(self)
            self.orig = self.overrideAttr(obj, 'test_attr')

        def test_value(self):
            self.assertEqual(tests._unitialized_attr, self.orig)
            self.assertRaises(AttributeError, getattr, obj, 'test_attr')
    self._run_successful_test(Test('test_value'))
    self.assertRaises(AttributeError, getattr, obj, 'test_attr')