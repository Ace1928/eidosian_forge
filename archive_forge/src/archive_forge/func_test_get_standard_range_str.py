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
def test_get_standard_range_str(self):
    result = self.driver1._get_standard_range_str(0, 5)
    self.assertEqual(result, 'bytes=0-4')
    result = self.driver1._get_standard_range_str(0)
    self.assertEqual(result, 'bytes=0-')
    result = self.driver1._get_standard_range_str(0, 1)
    self.assertEqual(result, 'bytes=0-0')
    result = self.driver1._get_standard_range_str(200)
    self.assertEqual(result, 'bytes=200-')
    result = self.driver1._get_standard_range_str(10, 200)
    self.assertEqual(result, 'bytes=10-199')
    result = self.driver1._get_standard_range_str(10, 11)
    self.assertEqual(result, 'bytes=10-10')
    result = self.driver1._get_standard_range_str(10, 11, True)
    self.assertEqual(result, 'bytes=10-11')