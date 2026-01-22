import json
import logging
from unittest import mock
import ddt
import fixtures
from keystoneauth1 import adapter
from keystoneauth1 import exceptions as keystone_exception
from oslo_serialization import jsonutils
from cinderclient import api_versions
import cinderclient.client
from cinderclient import exceptions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
@mock.patch.object(adapter.Adapter, 'request')
def test_sessionclient_request_method_raises_badrequest(self, mock_request):
    kwargs = {'body': {'volume': {'status': 'creating', 'imageRef': 'username', 'attach_status': 'detached'}, 'authenticated': 'True'}}
    resp = {'badRequest': {'message': 'Invalid image identifier or unable to access requested image.', 'code': 400}}
    mock_response = utils.TestResponse({'status_code': 400, 'text': json.dumps(resp).encode('latin-1')})
    mock_request.return_value = mock_response
    session_client = cinderclient.client.SessionClient(session=mock.Mock())
    self.assertRaises(exceptions.BadRequest, session_client.request, mock.sentinel.url, 'POST', **kwargs)
    self.assertIsNotNone(session_client._logger)