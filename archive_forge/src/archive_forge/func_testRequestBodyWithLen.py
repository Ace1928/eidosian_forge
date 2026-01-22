import socket
import unittest
import httplib2
from six.moves import http_client
from mock import patch
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
def testRequestBodyWithLen(self):
    http_wrapper.Request(body='burrito')