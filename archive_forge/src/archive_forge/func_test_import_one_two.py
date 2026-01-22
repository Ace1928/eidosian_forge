import linecache
import os
import re
import sys
from .. import errors, lazy_import, osutils
from . import TestCase, TestCaseInTempDir
import %(root_name)s.%(sub_name)s.%(submoda_name)s as submoda7
def test_import_one_two(self):
    exp = {'one': (['one'], None, {'two': (['one', 'two'], None, {})})}
    self.check(exp, 'import one.two')
    self.check(exp, 'import one, one.two')
    self.check(exp, 'import one\nimport one.two')