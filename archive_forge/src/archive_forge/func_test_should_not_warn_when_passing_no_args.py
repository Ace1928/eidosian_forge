from unittest import mock
import warnings
from oslotest import base
from monascaclient import client
@mock.patch('monascaclient.client.migration')
@mock.patch('monascaclient.client._get_auth_handler')
@mock.patch('monascaclient.client._get_session')
def test_should_not_warn_when_passing_no_args(self, _, __, ___):
    api_version = mock.Mock()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        client.Client(api_version)
        self.assertEqual(0, len(w))