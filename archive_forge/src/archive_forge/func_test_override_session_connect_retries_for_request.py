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
def test_override_session_connect_retries_for_request(self):
    session_retries = 1
    session = client_session.Session(connect_retries=session_retries)
    self.stub_url('GET', exc=requests.exceptions.Timeout())
    call_args = {'connect_retries': 0}
    with mock.patch('time.sleep') as m:
        self.assertRaises(exceptions.ConnectTimeout, session.request, self.TEST_URL, 'GET', **call_args)
        self.assertEqual(0, m.call_count)