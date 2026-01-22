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
@mock.patch('os.path.getsize')
def test_ex_multipart_upload_object_for_small_files(self, getsize_mock):
    getsize_mock.return_value = 0
    old_func = CloudFilesStorageDriver.upload_object
    mocked_upload_object = mock.Mock(return_value='test')
    CloudFilesStorageDriver.upload_object = mocked_upload_object
    file_path = os.path.abspath(__file__)
    container = Container(name='foo_bar_container', extra={}, driver=self)
    object_name = 'foo_test_upload'
    obj = self.driver.ex_multipart_upload_object(file_path=file_path, container=container, object_name=object_name)
    CloudFilesStorageDriver.upload_object = old_func
    self.assertTrue(mocked_upload_object.called)
    self.assertEqual(obj, 'test')