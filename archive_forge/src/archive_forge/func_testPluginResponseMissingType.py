import base64
import json
import struct
import sys
import mock
from pyu2f import errors
from pyu2f import model
from pyu2f.convenience import customauthenticator
@mock.patch.object(customauthenticator.subprocess, 'Popen')
@mock.patch.object(customauthenticator.os.environ, 'get', return_value='gnubbyagent')
def testPluginResponseMissingType(self, os_get_method, popen_method):
    """Test when plugin response is missing sign_helper_reply type."""
    valid_plugin_response = {'errorDetail': '', 'code': 0, 'responseData': {'appIdHash': SIGN_SUCCESS['app_id_hash_encoded'], 'challengeHash': SIGN_SUCCESS['challenge_hash_encoded'], 'keyHandle': SIGN_SUCCESS['key_handle_encoded'], 'version': SIGN_SUCCESS['u2f_version'], 'signatureData': SIGN_SUCCESS['signature_data_encoded']}, 'data': None}
    plugin_response_json = json.dumps(valid_plugin_response).encode()
    plugin_response_len = struct.pack('<I', len(plugin_response_json))
    mock_communicate_method = mock.MagicMock()
    mock_communicate_method.return_value = [plugin_response_len + plugin_response_json]
    mock_wait_method = mock.MagicMock()
    mock_wait_method.return_value = 0
    process_mock = mock.MagicMock()
    process_mock.communicate = mock_communicate_method
    process_mock.wait = mock_wait_method
    popen_method.return_value = process_mock
    challenge_data = [{'key': SIGN_SUCCESS['registered_key'], 'challenge': SIGN_SUCCESS['challenge']}]
    authenticator = customauthenticator.CustomAuthenticator(SIGN_SUCCESS['origin'])
    with self.assertRaises(errors.PluginError):
        authenticator.Authenticate(SIGN_SUCCESS['app_id'], challenge_data)