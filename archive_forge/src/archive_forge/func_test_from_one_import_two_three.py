import linecache
import os
import re
import sys
from .. import errors, lazy_import, osutils
from . import TestCase, TestCaseInTempDir
import %(root_name)s.%(sub_name)s.%(submoda_name)s as submoda7
def test_from_one_import_two_three(self):
    two_three_map = {'two': (['one'], 'two', {}), 'three': (['one'], 'three', {})}
    self.check_result(two_three_map, ['from one import two, three'])
    self.check_result(two_three_map, ['from one import two', 'from one import three'])