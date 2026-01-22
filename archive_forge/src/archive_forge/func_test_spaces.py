import os
import sys
import tempfile
from .. import mergetools, tests
def test_spaces(self):
    cmd_list = ['kdiff3', '{base}', '{this}', '{other}', '-o', '{result}']
    args, tmpfile = mergetools._subst_filename(cmd_list, 'file with space.txt')
    self.assertEqual(['kdiff3', 'file with space.txt.BASE', 'file with space.txt.THIS', 'file with space.txt.OTHER', '-o', 'file with space.txt'], args)