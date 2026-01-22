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
def test_unicode__walkdirs_utf8(self):
    """Walkdirs_utf8 should always return utf8 paths.

        The abspath portion might be in unicode or utf-8
        """
    self.requireFeature(features.UnicodeFilenameFeature)
    name0 = '0file-¶'
    name1 = '1dir-جو'
    name2 = '2file-س'
    tree = [name0, name1 + '/', name1 + '/' + name0, name1 + '/' + name1 + '/', name2]
    self.build_tree(tree)
    name0 = name0.encode('utf8')
    name1 = name1.encode('utf8')
    name2 = name2.encode('utf8')
    expected_dirblocks = [((b'', b'.'), [(name0, name0, 'file', b'./' + name0), (name1, name1, 'directory', b'./' + name1), (name2, name2, 'file', b'./' + name2)]), ((name1, b'./' + name1), [(name1 + b'/' + name0, name0, 'file', b'./' + name1 + b'/' + name0), (name1 + b'/' + name1, name1, 'directory', b'./' + name1 + b'/' + name1)]), ((name1 + b'/' + name1, b'./' + name1 + b'/' + name1), [])]
    result = []
    for dirdetail, dirblock in osutils._walkdirs_utf8('.'):
        self.assertIsInstance(dirdetail[0], bytes)
        if isinstance(dirdetail[1], str):
            dirdetail = (dirdetail[0], dirdetail[1].encode('utf8'))
            dirblock = [list(info) for info in dirblock]
            for info in dirblock:
                self.assertIsInstance(info[4], str)
                info[4] = info[4].encode('utf8')
        new_dirblock = []
        for info in dirblock:
            self.assertIsInstance(info[0], bytes)
            self.assertIsInstance(info[1], bytes)
            self.assertIsInstance(info[4], bytes)
            new_dirblock.append((info[0], info[1], info[2], info[4]))
        result.append((dirdetail, new_dirblock))
    self.assertEqual(expected_dirblocks, result)