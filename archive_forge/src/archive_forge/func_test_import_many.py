import linecache
import os
import re
import sys
from .. import errors, lazy_import, osutils
from . import TestCase, TestCaseInTempDir
import %(root_name)s.%(sub_name)s.%(submoda_name)s as submoda7
def test_import_many(self):
    exp = {'one': (['one'], None, {'two': (['one', 'two'], None, {'three': (['one', 'two', 'three'], None, {})}), 'four': (['one', 'four'], None, {})}), 'five': (['one', 'five'], None, {})}
    self.check(exp, 'import one.two.three, one.four, one.five as five')
    self.check(exp, 'import one.five as five\nimport one\nimport one.two.three\nimport one.four\n')