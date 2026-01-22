import datetime
import io
import itertools
import json
import logging
import sys
from unittest import mock
import uuid
from oslo_utils import encodeutils
import requests
import requests.auth
from testtools import matchers
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import plugin
from keystoneauth1 import session as client_session
from keystoneauth1.tests.unit import utils
from keystoneauth1 import token_endpoint
def test_additional_headers(self):
    session_key = uuid.uuid4().hex
    session_val = uuid.uuid4().hex
    adapter_key = uuid.uuid4().hex
    adapter_val = uuid.uuid4().hex
    request_key = uuid.uuid4().hex
    request_val = uuid.uuid4().hex
    text = uuid.uuid4().hex
    url = 'http://keystone.test.com'
    self.requests_mock.get(url, text=text)
    sess = client_session.Session(additional_headers={session_key: session_val})
    adap = adapter.Adapter(session=sess, additional_headers={adapter_key: adapter_val})
    resp = adap.get(url, headers={request_key: request_val})
    request = self.requests_mock.last_request
    self.assertEqual(resp.text, text)
    self.assertEqual(session_val, request.headers[session_key])
    self.assertEqual(adapter_val, request.headers[adapter_key])
    self.assertEqual(request_val, request.headers[request_key])