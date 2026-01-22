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
def test_upload_object_via_stream_illegal_seek_errors_are_ignored(self):
    size = 100
    self.driver1.connection = Mock()
    seek_error = OSError('Illegal seek')
    seek_error.errno = 29
    assert errno.ESPIPE == 29
    iterator = BodyStream('a' * size)
    iterator.seek = mock.Mock(side_effect=seek_error)
    result = self.driver1._upload_object(object_name='test1', content_type=None, request_path='/', stream=iterator)
    hasher = hashlib.md5()
    hasher.update(b('a') * size)
    expected_hash = hasher.hexdigest()
    self.assertEqual(result['data_hash'], expected_hash)
    self.assertEqual(result['bytes_transferred'], size)
    self.driver1.connection = Mock()
    seek_error = OSError('Other error')
    seek_error.errno = 21
    iterator = BodyStream('b' * size)
    iterator.seek = mock.Mock(side_effect=seek_error)
    self.assertRaisesRegex(OSError, 'Other error', self.driver1._upload_object, object_name='test1', content_type=None, request_path='/', stream=iterator)