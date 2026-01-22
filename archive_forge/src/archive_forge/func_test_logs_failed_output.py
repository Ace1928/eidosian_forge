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
def test_logs_failed_output(self):
    """Test that output is logged even for failed requests."""
    session = client_session.Session()
    body = {uuid.uuid4().hex: uuid.uuid4().hex}
    self.stub_url('GET', json=body, status_code=400, headers={'Content-Type': 'application/json'})
    resp = session.get(self.TEST_URL, raise_exc=False)
    self.assertEqual(resp.status_code, 400)
    self.assertIn(list(body.keys())[0], self.logger.output)
    self.assertIn(list(body.values())[0], self.logger.output)