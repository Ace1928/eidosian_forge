import base64
import datetime
import mock
import pytest  # type: ignore
import responses  # type: ignore
from google.auth import _helpers
from google.auth import exceptions
from google.auth import jwt
from google.auth import transport
from google.auth.compute_engine import credentials
from google.auth.transport import requests
@mock.patch('google.auth.compute_engine._metadata.get_service_account_info', autospec=True)
@mock.patch('google.auth.compute_engine._metadata.get', autospec=True)
def test_transport_error_from_metadata(self, get, get_service_account_info):
    get.side_effect = exceptions.TransportError('transport error')
    get_service_account_info.return_value = {'email': 'foo@example.com'}
    cred = credentials.IDTokenCredentials(mock.Mock(), 'audience', use_metadata_identity_endpoint=True)
    with pytest.raises(exceptions.RefreshError) as excinfo:
        cred.refresh(request=mock.Mock())
    assert excinfo.match('transport error')