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
def test_collect_timing(self):
    auth = AuthPlugin()
    sess = client_session.Session(auth=auth, collect_timing=True)
    response = {uuid.uuid4().hex: uuid.uuid4().hex}
    self.stub_url('GET', json=response, headers={'Content-Type': 'application/json'})
    resp = sess.get(self.TEST_URL)
    self.assertEqual(response, resp.json())
    timings = sess.get_timings()
    self.assertEqual(timings[0].method, 'GET')
    self.assertEqual(timings[0].url, self.TEST_URL)
    self.assertIsInstance(timings[0].elapsed, datetime.timedelta)
    sess.reset_timings()
    timings = sess.get_timings()
    self.assertEqual(len(timings), 0)