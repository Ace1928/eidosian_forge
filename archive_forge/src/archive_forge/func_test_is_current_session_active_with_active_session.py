import json
from unittest import mock
import ddt
from osprofiler.drivers import loginsight
from osprofiler import exc
from osprofiler.tests import test
@mock.patch.object(loginsight.LogInsightClient, '_send_request')
def test_is_current_session_active_with_active_session(self, send_request):
    self.assertTrue(self._client._is_current_session_active())
    exp_header = {'X-LI-Session-Id': self._client._session_id}
    send_request.assert_called_once_with('get', 'https', 'api/v1/sessions/current', headers=exp_header)