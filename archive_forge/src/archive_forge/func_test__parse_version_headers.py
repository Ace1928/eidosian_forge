from http import client as http_client
import json
import time
from unittest import mock
from keystoneauth1 import exceptions as kexc
from ironicclient.common import filecache
from ironicclient.common import http
from ironicclient import exc
from ironicclient.tests.unit import utils
def test__parse_version_headers(self):
    fake_session = utils.mockSession({'X-OpenStack-Ironic-API-Minimum-Version': '1.1', 'X-OpenStack-Ironic-API-Maximum-Version': '1.6', 'content-type': 'text/plain'}, None, http_client.HTTP_VERSION_NOT_SUPPORTED)
    expected_result = ('1.1', '1.6')
    client = _session_client(session=fake_session)
    result = client._parse_version_headers(fake_session.request())
    self.assertEqual(expected_result, result)