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
@mock.patch('libcloud.utils.files.exhaust_iterator')
@mock.patch('libcloud.utils.files.read_in_chunks')
def test_upload_object_hash_calculation_is_efficient(self, mock_read_in_chunks, mock_exhaust_iterator):
    size = 100
    self.driver1.connection = Mock()
    mock_read_in_chunks.return_value = 'a' * size
    iterator = BodyStream('a' * size)
    self.assertTrue(hasattr(iterator, '__next__'))
    self.assertTrue(hasattr(iterator, 'next'))
    self.assertEqual(mock_read_in_chunks.call_count, 0)
    self.assertEqual(mock_exhaust_iterator.call_count, 0)
    result = self.driver1._upload_object(object_name='test1', content_type=None, request_path='/', stream=iterator)
    hasher = hashlib.md5()
    hasher.update(b('a') * size)
    expected_hash = hasher.hexdigest()
    self.assertEqual(result['data_hash'], expected_hash)
    self.assertEqual(result['bytes_transferred'], size)
    headers = self.driver1.connection.request.call_args[-1]['headers']
    self.assertEqual(headers['Content-Type'], DEFAULT_CONTENT_TYPE)
    self.assertEqual(mock_read_in_chunks.call_count, 1)
    self.assertEqual(mock_exhaust_iterator.call_count, 0)
    mock_read_in_chunks.return_value = 'b' * size
    iterator = iter([str(v) for v in ['b' * size]])
    if PY2:
        self.assertFalse(hasattr(iterator, '__next__'))
        self.assertTrue(hasattr(iterator, 'next'))
    else:
        self.assertTrue(hasattr(iterator, '__next__'))
        self.assertFalse(hasattr(iterator, 'next'))
    self.assertEqual(mock_read_in_chunks.call_count, 1)
    self.assertEqual(mock_exhaust_iterator.call_count, 0)
    self.assertEqual(mock_read_in_chunks.call_count, 1)
    self.assertEqual(mock_exhaust_iterator.call_count, 0)
    result = self.driver1._upload_object(object_name='test2', content_type=None, request_path='/', stream=iterator)
    hasher = hashlib.md5()
    hasher.update(b('b') * size)
    expected_hash = hasher.hexdigest()
    self.assertEqual(result['data_hash'], expected_hash)
    self.assertEqual(result['bytes_transferred'], size)
    headers = self.driver1.connection.request.call_args[-1]['headers']
    self.assertEqual(headers['Content-Type'], DEFAULT_CONTENT_TYPE)
    self.assertEqual(mock_read_in_chunks.call_count, 2)
    self.assertEqual(mock_exhaust_iterator.call_count, 0)