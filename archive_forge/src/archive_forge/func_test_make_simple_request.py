from http import client as http_client
import json
import time
from unittest import mock
from keystoneauth1 import exceptions as kexc
from ironicclient.common import filecache
from ironicclient.common import http
from ironicclient import exc
from ironicclient.tests.unit import utils
def test_make_simple_request(self):
    session = utils.mockSession({})
    client = _session_client(session=session)
    res = client._make_simple_request(session, 'GET', 'url')
    session.request.assert_called_once_with('url', 'GET', raise_exc=False, endpoint_filter={'interface': 'publicURL', 'service_type': 'baremetal', 'region_name': ''}, endpoint_override='http://localhost:1234', user_agent=http.USER_AGENT)
    self.assertEqual(res, session.request.return_value)