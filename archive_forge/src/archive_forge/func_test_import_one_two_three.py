import linecache
import os
import re
import sys
from .. import errors, lazy_import, osutils
from . import TestCase, TestCaseInTempDir
import %(root_name)s.%(sub_name)s.%(submoda_name)s as submoda7
def test_import_one_two_three(self):
    one_two_three_map = {'one': (['one'], None, {'two': (['one', 'two'], None, {'three': (['one', 'two', 'three'], None, {})})})}
    self.check(one_two_three_map, ['import one.two.three'])
    self.check(one_two_three_map, ['import one, one.two.three'])
    self.check(one_two_three_map, ['import one', 'import one.two.three'])
    self.check(one_two_three_map, ['import one.two.three', 'import one'])