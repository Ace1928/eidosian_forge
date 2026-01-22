from unittest import mock
import warnings
from oslotest import base
from monascaclient import client
@mock.patch('monascaclient.client.migration')
@mock.patch('monascaclient.client._get_auth_handler')
@mock.patch('monascaclient.client._get_session')
def test_should_override_insecure_with_negated_verify(self, _, get_auth, __):
    api_version = mock.Mock()
    auth_val = mock.Mock()
    get_auth.return_value = auth_val
    for insecure in [True, False]:
        warnings.resetwarnings()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            client.Client(api_version, insecure=insecure)
            self.assertEqual(1, len(w))
            self.assertEqual(DeprecationWarning, w[0].category)
            self.assertRegex(str(w[0].message), 'Usage of insecure has been deprecated in favour of')
        get_auth.assert_called_once_with({'verify': not insecure})
        get_auth.reset_mock()