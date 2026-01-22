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
def test_session_debug_output_logs_openstack_request_id(self):
    """Test x-openstack-request-id is logged in debug logs."""

    def get_response(log=True):
        session = client_session.Session(verify=False)
        endpoint_filter = {'service_name': 'Identity'}
        headers = {'X-OpenStack-Request-Id': 'req-1234'}
        body = 'BODYRESPONSE'
        data = 'BODYDATA'
        all_headers = dict(itertools.chain(headers.items()))
        self.stub_url('POST', text=body, headers=all_headers)
        resp = session.post(self.TEST_URL, endpoint_filter=endpoint_filter, headers=all_headers, data=data, log=log)
        return resp
    resp = get_response(log=False)
    self.assertEqual(resp.status_code, 200)
    expected_log = 'POST call to Identity for %s used request id req-1234' % self.TEST_URL
    self.assertNotIn(expected_log, self.logger.output)
    resp = get_response()
    self.assertEqual(resp.status_code, 200)
    self.assertIn(expected_log, self.logger.output)