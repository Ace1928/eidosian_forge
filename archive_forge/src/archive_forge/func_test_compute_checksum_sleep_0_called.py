import errno
import hashlib
import json
import os
import shutil
import stat
import tempfile
import time
from unittest import mock
import uuid
import yaml
from oslotest import base as test_base
from oslo_utils import fileutils
def test_compute_checksum_sleep_0_called(self):
    path = fileutils.write_to_tempfile(self.content)
    self.assertTrue(os.path.exists(path))
    self.check_file_content(self.content, path)
    expected_checksum = hashlib.sha256()
    expected_checksum.update(self.content)
    with mock.patch.object(time, 'sleep') as sleep_mock:
        actual_checksum = fileutils.compute_file_checksum(path, read_chunksize=4)
    sleep_mock.assert_has_calls([mock.call(0)] * 3)
    self.assertEqual(3, sleep_mock.call_count)
    self.assertEqual(expected_checksum.hexdigest(), actual_checksum)