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
def test_additional_headers_overrides(self):
    header = uuid.uuid4().hex
    session_val = uuid.uuid4().hex
    adapter_val = uuid.uuid4().hex
    request_val = uuid.uuid4().hex
    url = 'http://keystone.test.com'
    self.requests_mock.get(url)
    sess = client_session.Session(additional_headers={header: session_val})
    adap = adapter.Adapter(session=sess)
    adap.get(url)
    self.assertEqual(session_val, self.requests_mock.last_request.headers[header])
    adap.additional_headers[header] = adapter_val
    adap.get(url)
    self.assertEqual(adapter_val, self.requests_mock.last_request.headers[header])
    adap.get(url, headers={header: request_val})
    self.assertEqual(request_val, self.requests_mock.last_request.headers[header])