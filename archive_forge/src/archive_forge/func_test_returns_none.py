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
def test_returns_none(self):
    self.overrideAttr(osutils, '_FILESYSTEM_FINDER', osutils.MtabFilesystemFinder([]))
    self.assertIs(osutils.get_fs_type('/home/jelmer/blah'), None)
    self.assertIs(osutils.get_fs_type(b'/home/jelmer/blah'), None)
    self.assertIs(osutils.get_fs_type('/home/jelmer'), None)