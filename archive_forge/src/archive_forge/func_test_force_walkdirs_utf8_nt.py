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
def test_force_walkdirs_utf8_nt(self):
    self.requireFeature(test__walkdirs_win32.win32_readdir_feature)
    self._save_platform_info()
    from .._walkdirs_win32 import Win32ReadDir
    self.assertDirReaderIs(Win32ReadDir, '.')