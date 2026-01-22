import abc
from oslo_utils import uuidutils
import osprofiler.profiler
import osprofiler.web
from requests_mock.contrib import fixture as mock_fixture
import testtools
from neutronclient import client
from neutronclient.common import exceptions
def test_request_success(self):
    headers = {'Accept': 'application/json', 'X-OpenStack-Request-ID': self.req_id}
    self.requests.register_uri(METHOD, URL, request_headers=headers)
    self.http.request(URL, METHOD)