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
def test_specified_size(self):
    """Request a transfer larger than the maximum block size and verify
        that the maximum read doesn't exceed the block_size."""
    self.assertTrue(self.test_data_len > self.block_size)
    from_file = file_utils.FakeReadFile(self.test_data)
    to_file = BytesIO()
    osutils.pumpfile(from_file, to_file, self.test_data_len, self.block_size)
    self.assertTrue(from_file.get_max_read_size() > 0)
    self.assertEqual(from_file.get_max_read_size(), self.block_size)
    self.assertEqual(from_file.get_read_count(), 3)
    response_data = to_file.getvalue()
    if response_data != self.test_data:
        message = 'Data not equal.  Expected %d bytes, received %d.'
        self.fail(message % (len(response_data), self.test_data_len))