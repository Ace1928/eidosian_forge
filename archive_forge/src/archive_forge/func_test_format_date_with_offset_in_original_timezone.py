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
def test_format_date_with_offset_in_original_timezone(self):
    self.assertEqual('Thu 1970-01-01 00:00:00 +0000', osutils.format_date_with_offset_in_original_timezone(0))
    self.assertEqual('Fri 1970-01-02 03:46:40 +0000', osutils.format_date_with_offset_in_original_timezone(100000))
    self.assertEqual('Fri 1970-01-02 05:46:40 +0200', osutils.format_date_with_offset_in_original_timezone(100000, 7200))