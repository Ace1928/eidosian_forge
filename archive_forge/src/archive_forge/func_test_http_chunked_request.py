import functools
import json
import logging
from unittest import mock
import uuid
import fixtures
import io
from keystoneauth1 import session
from keystoneauth1 import token_endpoint
from oslo_utils import encodeutils
import requests
from requests_mock.contrib import fixture
from urllib import parse
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
import testtools
from testtools import matchers
import types
import glanceclient
from glanceclient.common import http
from glanceclient.tests import utils
def test_http_chunked_request(self):
    text = 'Ok'
    data = io.StringIO(text)
    path = '/v1/images/'
    self.mock.post(self.endpoint + path, text=text)
    headers = {'test': 'chunked_request'}
    resp, body = self.client.post(path, headers=headers, data=data)
    self.assertIsInstance(self.mock.last_request.body, types.GeneratorType)
    self.assertEqual(text, resp.text)