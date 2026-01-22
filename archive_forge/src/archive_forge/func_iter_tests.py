from __future__ import nested_scopes
import fnmatch
import os.path
from _pydev_runfiles.pydev_runfiles_coverage import start_coverage_support
from _pydevd_bundle.pydevd_constants import *  # @UnusedWildImport
import re
import time
def iter_tests(self, test_objs):
    import unittest
    tests = []
    for test_obj in test_objs:
        if isinstance(test_obj, unittest.TestSuite):
            tests.extend(self.iter_tests(test_obj._tests))
        elif isinstance(test_obj, unittest.TestCase):
            tests.append(test_obj)
    return tests