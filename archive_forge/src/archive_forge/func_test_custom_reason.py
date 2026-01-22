from hashlib import md5
import unittest
from tornado.escape import utf8
from tornado.testing import AsyncHTTPTestCase
from tornado.test import httpclient_test
from tornado.web import Application, RequestHandler
def test_custom_reason(self):
    response = self.fetch('/custom_reason')
    self.assertEqual(response.reason, 'Custom reason')