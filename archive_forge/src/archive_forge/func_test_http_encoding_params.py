import http.client
from unittest import mock
from oslo_log.fixture import logging_error as log_fixture
import testtools
from glance.common import auth
from glance.common import client
from glance.tests.unit import fixtures as glance_fixtures
from glance.tests import utils
@mock.patch.object(http.client.HTTPConnection, 'getresponse')
@mock.patch.object(http.client.HTTPConnection, 'request')
def test_http_encoding_params(self, _mock_req, _mock_resp):
    fake = utils.FakeHTTPResponse(data=b'Ok')
    _mock_resp.return_value = fake
    params = {'test': 'niño'}
    resp = self.client.do_request('GET', '/v1/images/detail', params=params)
    self.assertEqual(fake, resp)