import os
import sys
import time
import shutil
import platform
import tempfile
import unittest
import multiprocessing
from libcloud.utils.files import exhaust_iterator
from libcloud.common.types import LibcloudError
from libcloud.storage.base import Object, Container
from libcloud.storage.types import (
def test_download_object_range_invalid_values(self):
    obj = Object('a', 500, '', {}, {}, None, None)
    tmppath = self.make_tmp_file(content='')
    expected_msg = 'start_bytes must be greater than 0'
    self.assertRaisesRegex(ValueError, expected_msg, self.driver.download_object_range, obj=obj, destination_path=tmppath, start_bytes=-1)
    expected_msg = 'start_bytes must be smaller than end_bytes'
    self.assertRaisesRegex(ValueError, expected_msg, self.driver.download_object_range, obj=obj, destination_path=tmppath, start_bytes=5, end_bytes=4)
    expected_msg = "start_bytes and end_bytes can't be the same"
    self.assertRaisesRegex(ValueError, expected_msg, self.driver.download_object_range, obj=obj, destination_path=tmppath, start_bytes=5, end_bytes=5)