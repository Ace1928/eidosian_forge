import abc
from oslo_utils import uuidutils
import osprofiler.profiler
import osprofiler.web
from requests_mock.contrib import fixture as mock_fixture
import testtools
from neutronclient import client
from neutronclient.common import exceptions
def test_osprofiler_headers_are_injected(self):
    osprofiler.profiler.init('SWORDFISH')
    self.addCleanup(osprofiler.profiler.clean)
    headers = {'Accept': 'application/json'}
    headers.update(osprofiler.web.get_trace_id_headers())
    self._test_headers(headers)