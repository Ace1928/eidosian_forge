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
def test__walkdirs_utf8_with_unicode_fs(self):
    """UnicodeDirReader should be a safe fallback everywhere

        The abspath portion should be in unicode
        """
    self.requireFeature(features.UnicodeFilenameFeature)
    self._save_platform_info()
    osutils._selected_dir_reader = osutils.UnicodeDirReader()
    name0u = '0file-¶'
    name1u = '1dir-جو'
    name2u = '2file-س'
    tree = [name0u, name1u + '/', name1u + '/' + name0u, name1u + '/' + name1u + '/', name2u]
    self.build_tree(tree)
    name0 = name0u.encode('utf8')
    name1 = name1u.encode('utf8')
    name2 = name2u.encode('utf8')
    expected_dirblocks = [((b'', '.'), [(name0, name0, 'file', './' + name0u), (name1, name1, 'directory', './' + name1u), (name2, name2, 'file', './' + name2u)]), ((name1, './' + name1u), [(name1 + b'/' + name0, name0, 'file', './' + name1u + '/' + name0u), (name1 + b'/' + name1, name1, 'directory', './' + name1u + '/' + name1u)]), ((name1 + b'/' + name1, './' + name1u + '/' + name1u), [])]
    result = list(osutils._walkdirs_utf8('.'))
    self._filter_out_stat(result)
    self.assertEqual(expected_dirblocks, result)