import os
import sys
import hmac
import base64
import tempfile
from io import BytesIO
from hashlib import sha1
from unittest import mock
from unittest.mock import Mock, PropertyMock
import libcloud.utils.files  # NOQA: F401
from libcloud.test import MockHttp  # pylint: disable-msg=E0611  # noqa
from libcloud.test import unittest, make_response, generate_random_data
from libcloud.utils.py3 import ET, StringIO, b, httplib, parse_qs, urlparse
from libcloud.utils.files import exhaust_iterator
from libcloud.common.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.storage.base import Object, Container
from libcloud.test.secrets import STORAGE_S3_PARAMS
from libcloud.storage.types import (
from libcloud.test.storage.base import BaseRangeDownloadMockHttp
from libcloud.storage.drivers.s3 import (
from libcloud.test.file_fixtures import StorageFileFixtures  # pylint: disable-msg=E0611
def test_download_object_as_stream_data_is_not_buffered_in_memory(self):
    mock_response = Mock(name='mock response')
    mock_response.headers = {}
    mock_response.status = 200
    msg1 = '"response" attribute was accessed but it shouldn\'t have been'
    msg2 = '"content" attribute was accessed but it shouldn\'t have been'
    type(mock_response).response = PropertyMock(name='mock response attribute', side_effect=Exception(msg1))
    type(mock_response).content = PropertyMock(name='mock content attribute', side_effect=Exception(msg2))
    mock_response.iter_content.return_value = StringIO('a' * 1000)
    self.driver.connection.request = Mock()
    self.driver.connection.request.return_value = mock_response
    container = Container(name='foo_bar_container', extra={}, driver=self.driver)
    obj = Object(name='foo_bar_object_NO_BUFFER', size=1000, hash=None, extra={}, container=container, meta_data=None, driver=self.driver_type)
    result = self.driver.download_object_as_stream(obj=obj)
    result = exhaust_iterator(result)
    result = result.decode('utf-8')
    self.assertEqual(result, 'a' * 1000)