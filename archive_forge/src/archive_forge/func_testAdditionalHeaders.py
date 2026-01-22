import base64
import datetime
import sys
import contextlib
import unittest
import six
from six.moves import http_client
from six.moves import urllib_parse
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.py import base_api
from apitools.base.py import encoding
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
def testAdditionalHeaders(self):
    additional_headers = {'Request-Is-Awesome': '1'}
    client = self.__GetFakeClient()
    http_request = http_wrapper.Request('http://www.example.com')
    new_request = client.ProcessHttpRequest(http_request)
    self.assertFalse('Request-Is-Awesome' in new_request.headers)
    client.additional_http_headers = additional_headers
    http_request = http_wrapper.Request('http://www.example.com')
    new_request = client.ProcessHttpRequest(http_request)
    self.assertTrue('Request-Is-Awesome' in new_request.headers)