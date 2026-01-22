import linecache
import os
import re
import sys
from .. import errors, lazy_import, osutils
from . import TestCase, TestCaseInTempDir
import %(root_name)s.%(sub_name)s.%(submoda_name)s as submoda7
def test_import_as(self):
    exp = {'two': (['one'], None, {})}
    self.check(exp, 'import one as two')