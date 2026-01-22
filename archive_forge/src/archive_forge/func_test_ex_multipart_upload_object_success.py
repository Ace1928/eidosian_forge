import os
import sys
import copy
import hmac
import math
import hashlib
import os.path  # pylint: disable-msg=W0404
from io import BytesIO
from hashlib import sha1
from unittest import mock
from unittest.mock import Mock, PropertyMock
import libcloud.utils.files
from libcloud.test import MockHttp  # pylint: disable-msg=E0611
from libcloud.test import unittest, make_response, generate_random_data
from libcloud.utils.py3 import StringIO, b, httplib, urlquote
from libcloud.utils.files import exhaust_iterator
from libcloud.common.types import MalformedResponseError
from libcloud.storage.base import CHUNK_SIZE, Object, Container
from libcloud.storage.types import (
from libcloud.test.storage.base import BaseRangeDownloadMockHttp
from libcloud.test.file_fixtures import StorageFileFixtures  # pylint: disable-msg=E0611
from libcloud.storage.drivers.cloudfiles import CloudFilesStorageDriver
def test_ex_multipart_upload_object_success(self):
    _upload_object_part = CloudFilesStorageDriver._upload_object_part
    _upload_object_manifest = CloudFilesStorageDriver._upload_object_manifest
    mocked__upload_object_part = mock.Mock(return_value='test_part')
    mocked__upload_object_manifest = mock.Mock(return_value='test_manifest')
    CloudFilesStorageDriver._upload_object_part = mocked__upload_object_part
    CloudFilesStorageDriver._upload_object_manifest = mocked__upload_object_manifest
    parts = 5
    file_path = os.path.abspath(__file__)
    chunk_size = int(math.ceil(float(os.path.getsize(file_path)) / parts))
    container = Container(name='foo_bar_container', extra={}, driver=self)
    object_name = 'foo_test_upload'
    self.driver.ex_multipart_upload_object(file_path=file_path, container=container, object_name=object_name, chunk_size=chunk_size)
    CloudFilesStorageDriver._upload_object_part = _upload_object_part
    CloudFilesStorageDriver._upload_object_manifest = _upload_object_manifest
    self.assertEqual(mocked__upload_object_part.call_count, parts)
    self.assertTrue(mocked__upload_object_manifest.call_count, 1)