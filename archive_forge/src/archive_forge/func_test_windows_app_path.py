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
def test_windows_app_path(self):
    if sys.platform != 'win32':
        raise tests.TestSkipped('test requires win32')
    self.overrideEnv('PATH', '')
    self.assertTrue(osutils.find_executable_on_path('iexplore') is not None)