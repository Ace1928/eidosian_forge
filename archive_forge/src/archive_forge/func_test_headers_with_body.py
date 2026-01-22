import abc
from oslo_utils import uuidutils
import osprofiler.profiler
import osprofiler.web
from requests_mock.contrib import fixture as mock_fixture
import testtools
from neutronclient import client
from neutronclient.common import exceptions
def test_headers_with_body(self):
    headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
    self._test_headers(headers, body=BODY)