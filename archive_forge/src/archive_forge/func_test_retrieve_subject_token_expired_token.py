import json
import os
import subprocess
import mock
import pytest  # type: ignore
from google.auth import exceptions
from google.auth import pluggable
from tests.test__default import WORKFORCE_AUDIENCE
@mock.patch.dict(os.environ, {'GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES': '1'})
def test_retrieve_subject_token_expired_token(self):
    EXECUTABLE_SUCCESSFUL_OIDC_RESPONSE_EXPIRED = {'version': 1, 'success': True, 'token_type': 'urn:ietf:params:oauth:token-type:id_token', 'id_token': self.EXECUTABLE_OIDC_TOKEN, 'expiration_time': 0}
    with mock.patch('subprocess.run', return_value=subprocess.CompletedProcess(args=[], stdout=json.dumps(EXECUTABLE_SUCCESSFUL_OIDC_RESPONSE_EXPIRED).encode('UTF-8'), returncode=0)):
        credentials = self.make_pluggable(credential_source=self.CREDENTIAL_SOURCE)
        with pytest.raises(exceptions.RefreshError) as excinfo:
            _ = credentials.retrieve_subject_token(None)
        assert excinfo.match('The token returned by the executable is expired.')