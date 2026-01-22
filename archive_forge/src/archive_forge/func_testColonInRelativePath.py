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
def testColonInRelativePath(self):
    method_config = base_api.ApiMethodInfo(relative_path='path:withJustColon', request_type_name='SimpleMessage')
    service = FakeService()
    request = SimpleMessage()
    http_request = service.PrepareHttpRequest(method_config, request)
    self.assertEqual('http://www.example.com/path:withJustColon', http_request.url)