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
def test_applyDeprecated_not_deprecated(self):
    sample_object = ApplyDeprecatedHelper()
    self.assertRaises(AssertionError, self.applyDeprecated, deprecated_in((0, 11, 0)), sample_object.sample_normal_method)
    self.assertRaises(AssertionError, self.applyDeprecated, deprecated_in((0, 11, 0)), sample_undeprecated_function, 'a param value')
    self.assertRaises(AssertionError, self.applyDeprecated, deprecated_in((0, 10, 0)), sample_object.sample_deprecated_method, 'a param value')
    self.assertRaises(AssertionError, self.applyDeprecated, deprecated_in((0, 10, 0)), sample_deprecated_function)
    self.assertEqual('a param value', self.applyDeprecated(deprecated_in((0, 11, 0)), sample_object.sample_deprecated_method, 'a param value'))
    self.assertEqual(2, self.applyDeprecated(deprecated_in((0, 11, 0)), sample_deprecated_function))
    self.assertRaises(AssertionError, self.applyDeprecated, deprecated_in((0, 11, 0)), sample_object.sample_nested_deprecation)
    self.assertEqual(2, self.applyDeprecated(deprecated_in((0, 10, 0)), sample_object.sample_nested_deprecation))