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
def testIncludeEmptyFieldsClient(self):
    msg = SimpleMessage()
    client = self.__GetFakeClient()
    self.assertEqual('{}', client.SerializeMessage(msg))
    with client.IncludeFields(('field',)):
        self.assertEqual('{"field": null}', client.SerializeMessage(msg))