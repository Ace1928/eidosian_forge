import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import b, httplib, urlparse, parse_qsl
from libcloud.common.types import MalformedResponseError
from libcloud.common.cloudstack import CloudStackConnection
def test_async_request_unsuccessful(self):
    self.driver.path = '/async/fail'
    try:
        self.connection._async_request('fake')
    except Exception as e:
        self.assertEqual(CloudStackMockHttp.ERROR_TEXT, str(e))
        return
    self.assertFalse(True)