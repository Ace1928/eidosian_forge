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
def testQueryBytesRequest(self):
    method_config = base_api.ApiMethodInfo(request_type_name='SimpleMessage', query_params=['bytes_field'])
    service = FakeService()
    non_unicode_message = b''.join((six.int2byte(100), six.int2byte(200)))
    request = SimpleMessage(bytes_field=non_unicode_message)
    global_params = StandardQueryParameters()
    http_request = service.PrepareHttpRequest(method_config, request, global_params=global_params)
    want = urllib_parse.urlencode({'bytes_field': base64.urlsafe_b64encode(non_unicode_message)})
    self.assertIn(want, http_request.url)