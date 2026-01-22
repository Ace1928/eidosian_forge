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
def test_compute_checksum_named_algorithm(self):
    path = fileutils.write_to_tempfile(self.content)
    self.assertTrue(os.path.exists(path))
    self.check_file_content(self.content, path)
    expected_checksum = hashlib.sha512()
    expected_checksum.update(self.content)
    actual_checksum = fileutils.compute_file_checksum(path, algorithm='sha512')
    self.assertEqual(expected_checksum.hexdigest(), actual_checksum)