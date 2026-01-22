import linecache
import os
import re
import sys
from .. import errors, lazy_import, osutils
from . import TestCase, TestCaseInTempDir
import %(root_name)s.%(sub_name)s.%(submoda_name)s as submoda7
def test_import_one_two_as_x(self):
    self.check({'x': (['one', 'two'], None, {})}, ['import one.two as x'])