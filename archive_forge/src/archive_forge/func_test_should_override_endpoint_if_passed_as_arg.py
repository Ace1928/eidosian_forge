from unittest import mock
import warnings
from oslotest import base
from monascaclient import client
@mock.patch('monascaclient.client.migration')
@mock.patch('monascaclient.client._get_auth_handler')
@mock.patch('monascaclient.client._get_session')
def test_should_override_endpoint_if_passed_as_arg(self, get_session, get_auth, _):
    api_version = mock.Mock()
    endpoint = mock.Mock()
    endpoint_fake = mock.Mock()
    auth_val = mock.Mock()
    get_auth.return_value = auth_val
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        client.Client(api_version, endpoint, endpoint=endpoint_fake)
        self.assertEqual(1, len(w))
    get_auth.assert_called_once_with({'endpoint': endpoint})
    get_session.assert_called_once_with(auth_val, {'endpoint': endpoint})