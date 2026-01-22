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
def test_adapter_connect_retries(self):
    retries = 2
    sess = client_session.Session()
    adpt = adapter.Adapter(sess, connect_retries=retries)
    self.stub_url('GET', exc=requests.exceptions.ConnectionError())
    with mock.patch('time.sleep') as m:
        self.assertRaises(exceptions.ConnectionError, adpt.get, self.TEST_URL)
        self.assertEqual(retries, m.call_count)
    self.assertThat(self.requests_mock.request_history, matchers.HasLength(retries + 1))