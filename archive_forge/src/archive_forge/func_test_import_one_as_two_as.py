import linecache
import os
import re
import sys
from .. import errors, lazy_import, osutils
from . import TestCase, TestCaseInTempDir
import %(root_name)s.%(sub_name)s.%(submoda_name)s as submoda7
def test_import_one_as_two_as(self):
    self.check(['import one as x, two as y'], 'import one as x, two as y')
    self.check(['import one as x, two as y'], '\nimport one as x, two as y\n')