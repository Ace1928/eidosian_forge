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
def test__upload_object_manifest_wrong_hash(self):
    fake_response = type('CloudFilesResponse', (), {'headers': {'etag': '0000000'}})
    _request = self.driver.connection.request
    mocked_request = mock.Mock(return_value=fake_response)
    self.driver.connection.request = mocked_request
    container = Container(name='foo_bar_container', extra={}, driver=self)
    object_name = 'test_object'
    try:
        self.driver._upload_object_manifest(container, object_name)
    except ObjectHashMismatchError:
        pass
    else:
        self.fail('Exception was not thrown')
    finally:
        self.driver.connection.request = _request