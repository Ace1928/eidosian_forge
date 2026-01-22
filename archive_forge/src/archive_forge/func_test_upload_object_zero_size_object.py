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
def test_upload_object_zero_size_object(self):

    def upload_file(self, object_name=None, content_type=None, request_path=None, request_method=None, headers=None, file_path=None, stream=None):
        return {'response': make_response(201, headers={'etag': '0cc175b9c0f1b6a831c399e269772661'}), 'bytes_transferred': 0, 'data_hash': '0cc175b9c0f1b6a831c399e269772661'}
    old_func = CloudFilesStorageDriver._upload_object
    CloudFilesStorageDriver._upload_object = upload_file
    old_request = self.driver.connection.request
    file_path = os.path.join(os.path.dirname(__file__), '__init__.py')
    container = Container(name='foo_bar_container', extra={}, driver=self)
    object_name = 'empty'
    extra = {}

    def func(*args, **kwargs):
        self.assertEqual(kwargs['headers']['Content-Length'], 0)
        func.called = True
        return old_request(*args, **kwargs)
    self.driver.connection.request = func
    func.called = False
    obj = self.driver.upload_object(file_path=file_path, container=container, extra=extra, object_name=object_name)
    self.assertEqual(obj.name, 'empty')
    self.assertEqual(obj.size, 0)
    CloudFilesStorageDriver._upload_object = old_func
    self.driver.connection.request = old_request