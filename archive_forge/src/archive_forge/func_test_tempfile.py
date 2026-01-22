import os
import sys
import tempfile
from .. import mergetools, tests
def test_tempfile(self):
    self.build_tree(('test.txt', 'test.txt.BASE', 'test.txt.THIS', 'test.txt.OTHER'))
    cmd_list = ['some_tool', '{this_temp}']
    args, tmpfile = mergetools._subst_filename(cmd_list, 'test.txt')
    self.assertPathExists(tmpfile)
    os.remove(tmpfile)