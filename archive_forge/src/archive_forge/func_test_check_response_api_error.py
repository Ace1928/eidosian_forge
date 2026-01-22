import json
from unittest import mock
import ddt
from osprofiler.drivers import loginsight
from osprofiler import exc
from osprofiler.tests import test
def test_check_response_api_error(self):
    resp = mock.Mock(status_code=401, ok=False)
    resp.text = json.dumps({'errorMessage': 'Invalid username or password.', 'errorCode': 'FIELD_ERROR'})
    e = self.assertRaises(exc.LogInsightAPIError, self._client._check_response, resp)
    self.assertEqual('Invalid username or password.', str(e))