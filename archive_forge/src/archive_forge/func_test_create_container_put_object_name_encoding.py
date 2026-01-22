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
def test_create_container_put_object_name_encoding(self):

    def upload_file(self, object_name=None, content_type=None, request_path=None, request_method=None, headers=None, file_path=None, stream=None):
        return {'response': make_response(201, headers={'etag': '0cc175b9c0f1b6a831c399e269772661'}), 'bytes_transferred': 1000, 'data_hash': '0cc175b9c0f1b6a831c399e269772661'}
    old_func = CloudFilesStorageDriver._upload_object
    CloudFilesStorageDriver._upload_object = upload_file
    container_name = 'speci@l_name'
    object_name = 'm@obj€ct'
    file_path = os.path.abspath(__file__)
    container = self.driver.create_container(container_name=container_name)
    self.assertEqual(container.name, container_name)
    obj = self.driver.upload_object(file_path=file_path, container=container, object_name=object_name)
    self.assertEqual(obj.name, object_name)
    CloudFilesStorageDriver._upload_object = old_func