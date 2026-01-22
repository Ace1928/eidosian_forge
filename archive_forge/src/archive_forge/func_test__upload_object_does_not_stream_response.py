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
def test__upload_object_does_not_stream_response(self):
    resp = self.driver1._upload_object(object_name='foo', content_type='foo/bar', request_path='/', stream=iter(b'foo'))
    mock_response = resp['response'].response._response
    response_streamed = mock_response.request.stream
    assert response_streamed is False