from unittest import mock
import urllib.parse
import fixtures
from oslo_serialization import jsonutils
import requests
from requests_mock.contrib import fixture as rm_fixture
from testtools import matchers
import webob
from keystonemiddleware import s3_token
from keystonemiddleware.tests.unit import utils
def test_bad_reply(self):
    self.requests_mock.post(self.TEST_URL, status_code=201, text='<badreply>')
    req = webob.Request.blank('/v1/AUTH_cfa/c/o')
    req.headers['Authorization'] = 'access:signature'
    req.headers['X-Storage-Token'] = 'token'
    resp = req.get_response(self.middleware)
    s3_invalid_req = self.middleware._deny_request('InvalidURI')
    self.assertEqual(resp.body, s3_invalid_req.body)
    self.assertEqual(resp.status_int, s3_invalid_req.status_int)