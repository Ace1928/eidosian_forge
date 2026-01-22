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
def test_pathjoin(self):
    self.assertEqual('path/to/foo', osutils._win32_pathjoin('path', 'to', 'foo'))
    self.assertEqual('C:/foo', osutils._win32_pathjoin('path\\to', 'C:\\foo'))
    self.assertEqual('C:/foo', osutils._win32_pathjoin('path/to', 'C:/foo'))
    self.assertEqual('path/to/foo', osutils._win32_pathjoin('path/to/', 'foo'))