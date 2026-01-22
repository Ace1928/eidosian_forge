import base64
import sys
import mock
import pytest  # type: ignore
import pyu2f  # type: ignore
from google.auth import exceptions
from google.oauth2 import challenges
def test_security_key():
    metadata = {'status': 'READY', 'challengeId': 2, 'challengeType': 'SECURITY_KEY', 'securityKey': {'applicationId': 'security_key_application_id', 'challenges': [{'keyHandle': 'some_key', 'challenge': base64.urlsafe_b64encode('some_challenge'.encode('ascii')).decode('ascii')}], 'relyingPartyId': 'security_key_application_id'}}
    mock_key = mock.Mock()
    challenge = challenges.SecurityKeyChallenge()
    with mock.patch('pyu2f.model.RegisteredKey', return_value=mock_key):
        with mock.patch('pyu2f.convenience.authenticator.CompositeAuthenticator.Authenticate') as mock_authenticate:
            mock_authenticate.return_value = 'security key response'
            assert challenge.name == 'SECURITY_KEY'
            assert challenge.is_locally_eligible
            assert challenge.obtain_challenge_input(metadata) == {'securityKey': 'security key response'}
            mock_authenticate.assert_called_with('security_key_application_id', [{'key': mock_key, 'challenge': b'some_challenge'}], print_callback=sys.stderr.write)
    metadata['securityKey']['relyingPartyId'] = 'security_key_relying_party_id'
    sys.stderr.write('metadata=' + str(metadata) + '\n')
    with mock.patch('pyu2f.model.RegisteredKey', return_value=mock_key):
        with mock.patch('pyu2f.convenience.authenticator.CompositeAuthenticator.Authenticate') as mock_authenticate:
            mock_authenticate.return_value = 'security key response'
            assert challenge.name == 'SECURITY_KEY'
            assert challenge.is_locally_eligible
            assert challenge.obtain_challenge_input(metadata) == {'securityKey': 'security key response'}
            mock_authenticate.assert_called_with('security_key_relying_party_id', [{'key': mock_key, 'challenge': b'some_challenge'}], print_callback=sys.stderr.write)
    metadata['securityKey']['relyingPartyId'] = 'security_key_relying_party_id'
    with mock.patch('pyu2f.model.RegisteredKey', return_value=mock_key):
        with mock.patch('pyu2f.convenience.authenticator.CompositeAuthenticator.Authenticate') as mock_authenticate:
            assert challenge.name == 'SECURITY_KEY'
            assert challenge.is_locally_eligible
            mock_authenticate.side_effect = [pyu2f.errors.U2FError(pyu2f.errors.U2FError.DEVICE_INELIGIBLE), 'security key response']
            assert challenge.obtain_challenge_input(metadata) == {'securityKey': 'security key response'}
            calls = [mock.call('security_key_relying_party_id', [{'key': mock_key, 'challenge': b'some_challenge'}], print_callback=sys.stderr.write), mock.call('security_key_application_id', [{'key': mock_key, 'challenge': b'some_challenge'}], print_callback=sys.stderr.write)]
            mock_authenticate.assert_has_calls(calls)
    with mock.patch('pyu2f.model.RegisteredKey', return_value=mock_key):
        with mock.patch('pyu2f.convenience.authenticator.CompositeAuthenticator.Authenticate') as mock_authenticate:
            mock_authenticate.side_effect = pyu2f.errors.U2FError(pyu2f.errors.U2FError.DEVICE_INELIGIBLE)
            assert challenge.obtain_challenge_input(metadata) is None
        with mock.patch('pyu2f.convenience.authenticator.CompositeAuthenticator.Authenticate') as mock_authenticate:
            mock_authenticate.side_effect = pyu2f.errors.U2FError(pyu2f.errors.U2FError.TIMEOUT)
            assert challenge.obtain_challenge_input(metadata) is None
        with mock.patch('pyu2f.convenience.authenticator.CompositeAuthenticator.Authenticate') as mock_authenticate:
            mock_authenticate.side_effect = pyu2f.errors.PluginError()
            assert challenge.obtain_challenge_input(metadata) is None
        with mock.patch('pyu2f.convenience.authenticator.CompositeAuthenticator.Authenticate') as mock_authenticate:
            mock_authenticate.side_effect = pyu2f.errors.U2FError(pyu2f.errors.U2FError.BAD_REQUEST)
            with pytest.raises(pyu2f.errors.U2FError):
                challenge.obtain_challenge_input(metadata)
        with mock.patch('pyu2f.convenience.authenticator.CompositeAuthenticator.Authenticate') as mock_authenticate:
            mock_authenticate.side_effect = pyu2f.errors.NoDeviceFoundError()
            assert challenge.obtain_challenge_input(metadata) is None
        with mock.patch('pyu2f.convenience.authenticator.CompositeAuthenticator.Authenticate') as mock_authenticate:
            mock_authenticate.side_effect = pyu2f.errors.UnsupportedVersionException()
            with pytest.raises(pyu2f.errors.UnsupportedVersionException):
                challenge.obtain_challenge_input(metadata)
        with mock.patch.dict('sys.modules'):
            sys.modules['pyu2f'] = None
            with pytest.raises(exceptions.ReauthFailError) as excinfo:
                challenge.obtain_challenge_input(metadata)
            assert excinfo.match('pyu2f dependency is required')