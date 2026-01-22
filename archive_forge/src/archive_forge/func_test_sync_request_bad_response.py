import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import b, httplib, urlparse, parse_qsl
from libcloud.common.types import MalformedResponseError
from libcloud.common.cloudstack import CloudStackConnection
def test_sync_request_bad_response(self):
    self.driver.path = '/bad/response'
    try:
        self.connection._sync_request('fake')
    except Exception as e:
        self.assertTrue(isinstance(e, MalformedResponseError))
        return
    self.assertTrue(False)