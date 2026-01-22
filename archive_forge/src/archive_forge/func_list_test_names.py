from __future__ import nested_scopes
import fnmatch
import os.path
from _pydev_runfiles.pydev_runfiles_coverage import start_coverage_support
from _pydevd_bundle.pydevd_constants import *  # @UnusedWildImport
import re
import time
def list_test_names(self, test_objs):
    names = []
    for tc in self.iter_tests(test_objs):
        try:
            testMethodName = tc._TestCase__testMethodName
        except AttributeError:
            testMethodName = tc._testMethodName
        names.append(testMethodName)
    return names