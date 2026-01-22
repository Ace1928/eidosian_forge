import base64
import copy
from unittest import mock
from urllib import parse as urlparse
from oslo_utils import uuidutils
from osprofiler import _utils as osprofiler_utils
import osprofiler.profiler
from mistralclient.api import httpclient
from mistralclient.tests.unit import base
@mock.patch.object(osprofiler.profiler._Profiler, 'get_base_id', mock.MagicMock(return_value=PROFILER_TRACE_ID))
@mock.patch.object(osprofiler.profiler._Profiler, 'get_id', mock.MagicMock(return_value=PROFILER_TRACE_ID))
def test_get_request_options_with_profile_enabled(self):
    m = self.requests_mock.get(EXPECTED_URL, text='text')
    osprofiler.profiler.init(PROFILER_HMAC_KEY)
    data = {'base_id': PROFILER_TRACE_ID, 'parent_id': PROFILER_TRACE_ID}
    signed_data = osprofiler_utils.signed_pack(data, PROFILER_HMAC_KEY)
    headers = {'X-Trace-Info': signed_data[0], 'X-Trace-HMAC': signed_data[1]}
    self.client.get(API_URL)
    self.assertTrue(m.called_once)
    headers = self.assertExpectedAuthHeaders()
    self.assertEqual(signed_data[0], headers['X-Trace-Info'])
    self.assertEqual(signed_data[1], headers['X-Trace-HMAC'])