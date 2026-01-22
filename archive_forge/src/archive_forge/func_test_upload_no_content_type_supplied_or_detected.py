import sys
import errno
import hashlib
from io import BytesIO
from unittest import mock
from unittest.mock import Mock
from libcloud.test import MockHttp, BodyStream, unittest
from libcloud.utils.py3 import PY2, StringIO, b, httplib, assertRaisesRegex
from libcloud.storage.base import DEFAULT_CONTENT_TYPE, StorageDriver
from libcloud.common.exceptions import RateLimitReachedError
from libcloud.test.storage.base import BaseRangeDownloadMockHttp
def test_upload_no_content_type_supplied_or_detected(self):
    iterator = StringIO()
    self.driver1.connection = Mock()
    self.driver1._upload_object(object_name='test', content_type=None, request_path='/', stream=iterator)
    headers = self.driver1.connection.request.call_args[-1]['headers']
    self.assertEqual(headers['Content-Type'], DEFAULT_CONTENT_TYPE)
    self.driver1.strict_mode = True
    expected_msg = 'File content-type could not be guessed for "test" and no content_type value is provided'
    assertRaisesRegex(self, AttributeError, expected_msg, self.driver1._upload_object, object_name='test', content_type=None, request_path='/', stream=iterator)