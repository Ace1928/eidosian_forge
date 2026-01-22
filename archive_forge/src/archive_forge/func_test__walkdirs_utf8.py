import errno
import os
import select
import socket
import sys
import tempfile
import time
from io import BytesIO
from .. import errors, osutils, tests, trace, win32utils
from . import features, file_utils, test__walkdirs_win32
from .scenarios import load_tests_apply_scenarios
def test__walkdirs_utf8(self):
    tree = ['.bzr', '0file', '1dir/', '1dir/0file', '1dir/1dir/', '2file']
    self.build_tree(tree)
    expected_dirblocks = [(('', '.'), [('0file', '0file', 'file'), ('1dir', '1dir', 'directory'), ('2file', '2file', 'file')]), (('1dir', './1dir'), [('1dir/0file', '0file', 'file'), ('1dir/1dir', '1dir', 'directory')]), (('1dir/1dir', './1dir/1dir'), [])]
    result = []
    found_bzrdir = False
    for dirdetail, dirblock in osutils._walkdirs_utf8(b'.'):
        if len(dirblock) and dirblock[0][1] == b'.bzr':
            found_bzrdir = True
            del dirblock[0]
        dirdetail = (dirdetail[0].decode('utf-8'), osutils.safe_unicode(dirdetail[1]))
        dirblock = [(entry[0].decode('utf-8'), entry[1].decode('utf-8'), entry[2]) for entry in dirblock]
        result.append((dirdetail, dirblock))
    self.assertTrue(found_bzrdir)
    self.assertExpectedBlocks(expected_dirblocks, result)
    result = []
    for dirblock in osutils.walkdirs('./1dir', '1dir'):
        result.append(dirblock)
    self.assertExpectedBlocks(expected_dirblocks[1:], result)