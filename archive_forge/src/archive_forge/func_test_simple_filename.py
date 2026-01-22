import os
import sys
import tempfile
from .. import mergetools, tests
def test_simple_filename(self):
    cmd_list = ['kdiff3', '{base}', '{this}', '{other}', '-o', '{result}']
    args, tmpfile = mergetools._subst_filename(cmd_list, 'test.txt')
    self.assertEqual(['kdiff3', 'test.txt.BASE', 'test.txt.THIS', 'test.txt.OTHER', '-o', 'test.txt'], args)