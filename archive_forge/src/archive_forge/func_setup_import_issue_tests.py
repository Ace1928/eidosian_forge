import os.path
from os.path import abspath
import re
import sys
import types
import pickle
from test import support
from test.support import import_helper
import unittest
import unittest.mock
import unittest.test
def setup_import_issue_tests(self, fakefile):
    listdir = os.listdir
    os.listdir = lambda _: [fakefile]
    isfile = os.path.isfile
    os.path.isfile = lambda _: True
    orig_sys_path = sys.path[:]

    def restore():
        os.path.isfile = isfile
        os.listdir = listdir
        sys.path[:] = orig_sys_path
    self.addCleanup(restore)