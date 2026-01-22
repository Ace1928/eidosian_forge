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
def test_download_object_range_as_stream_invalid_values(self):
    content = b'0123456789123456789'
    tmppath = self.make_tmp_file(content=content)
    container = self.driver.create_container('test6')
    obj = container.upload_object(tmppath, 'test')
    expected_msg = 'start_bytes must be greater than 0'
    stream = self.driver.download_object_range_as_stream(obj=obj, start_bytes=-1, end_bytes=None, chunk_size=1024)
    self.assertRaisesRegex(ValueError, expected_msg, exhaust_iterator, stream)
    expected_msg = 'start_bytes must be smaller than end_bytes'
    stream = self.driver.download_object_range_as_stream(obj=obj, start_bytes=5, end_bytes=4, chunk_size=1024)
    self.assertRaisesRegex(ValueError, expected_msg, exhaust_iterator, stream)
    expected_msg = 'end_bytes is larger than file size'
    stream = self.driver.download_object_range_as_stream(obj=obj, start_bytes=5, end_bytes=len(content) + 1, chunk_size=1024)
    expected_msg = "start_bytes and end_bytes can't be the same"
    stream = self.driver.download_object_range_as_stream(obj=obj, start_bytes=5, end_bytes=5, chunk_size=1024)
    obj.delete()
    container.delete()
    self.remove_tmp_file(tmppath)