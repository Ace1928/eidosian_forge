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
def testQueryEncoding(self):
    method_config = base_api.ApiMethodInfo(request_type_name='MessageWithTime', query_params=['timestamp'])
    service = FakeService()
    request = MessageWithTime(timestamp=datetime.datetime(2014, 10, 7, 12, 53, 13))
    http_request = service.PrepareHttpRequest(method_config, request)
    url_timestamp = urllib_parse.quote(request.timestamp.isoformat())
    self.assertTrue(http_request.url.endswith(url_timestamp))