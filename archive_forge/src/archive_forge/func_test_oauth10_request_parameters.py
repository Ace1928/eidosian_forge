import unittest
from tornado.auth import (
from tornado.escape import json_decode
from tornado import gen
from tornado.httpclient import HTTPClientError
from tornado.httputil import url_concat
from tornado.log import app_log
from tornado.testing import AsyncHTTPTestCase, ExpectLog
from tornado.web import RequestHandler, Application, HTTPError
def test_oauth10_request_parameters(self):
    response = self.fetch('/oauth10/client/request_params')
    response.rethrow()
    parsed = json_decode(response.body)
    self.assertEqual(parsed['oauth_consumer_key'], 'asdf')
    self.assertEqual(parsed['oauth_token'], 'uiop')
    self.assertTrue('oauth_nonce' in parsed)
    self.assertTrue('oauth_signature' in parsed)