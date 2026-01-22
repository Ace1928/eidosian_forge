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
def test_without_auth_storage_token(self):
    req = webob.Request.blank('/v1/AUTH_cfa/c/o')
    req.headers['Authorization'] = 'badboy'
    self.middleware(req.environ, self.start_fake_response)
    self.assertEqual(self.response_status, 200)