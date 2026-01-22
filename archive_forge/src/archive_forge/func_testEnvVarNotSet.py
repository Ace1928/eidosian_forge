import base64
import json
import struct
import sys
import mock
from pyu2f import errors
from pyu2f import model
from pyu2f.convenience import customauthenticator
@mock.patch.object(customauthenticator.os.environ, 'get', return_value=None)
def testEnvVarNotSet(self, os_get_method):
    authenticator = customauthenticator.CustomAuthenticator('testorigin')
    self.assertFalse(authenticator.IsAvailable(), 'Should return false when no env var is present')
    with self.assertRaises(errors.PluginError):
        authenticator.Authenticate('appid', {})