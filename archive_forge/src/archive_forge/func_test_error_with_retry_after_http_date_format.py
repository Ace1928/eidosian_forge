import sys
import unittest
from unittest import mock
from libcloud.test import LibcloudTestCase
from libcloud.common.base import Response, LazyObject
from libcloud.common.exceptions import BaseHTTPError, RateLimitReachedError
@mock.patch('time.time', return_value=1231006505)
def test_error_with_retry_after_http_date_format(self, time_mock):
    retry_after = 'Sat, 03 Jan 2009 18:20:05 -0000'
    resp_mock = self.mock_response(503, {'Retry-After': retry_after})
    try:
        Response(resp_mock, mock.MagicMock())
    except BaseHTTPError as e:
        self.assertIn('retry-after', e.headers)
        self.assertEqual(e.headers['retry-after'], '300')
    else:
        self.fail("HTTP Status 503 response didn't raised an exception")