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
def testCustomRetryFunc(self):

    def retry_func():
        pass

    def fakeMakeRequest(*_, **kwargs):
        self.assertEqual(retry_func, kwargs['retry_func'])
        return http_wrapper.Response(info={'status': '200'}, content='{"field": "abc"}', request_url='http://www.google.com')
    method_config = base_api.ApiMethodInfo(request_type_name='SimpleMessage', response_type_name='SimpleMessage')
    client = self.__GetFakeClient()
    client.retry_func = retry_func
    service = FakeService(client=client)
    request = SimpleMessage()
    with mock(base_api.http_wrapper, 'MakeRequest', fakeMakeRequest):
        service._RunMethod(method_config, request)