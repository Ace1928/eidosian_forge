import json
import os
import subprocess
import mock
import pytest  # type: ignore
from google.auth import exceptions
from google.auth import pluggable
from tests.test__default import WORKFORCE_AUDIENCE
@mock.patch.dict(os.environ, {'GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES': '1'})
def test_retrieve_subject_token_without_expiration_time_should_pass_when_output_file_not_specified(self):
    EXECUTABLE_SUCCESSFUL_OIDC_RESPONSE = {'version': 1, 'success': True, 'token_type': 'urn:ietf:params:oauth:token-type:id_token', 'id_token': self.EXECUTABLE_OIDC_TOKEN}
    CREDENTIAL_SOURCE = {'executable': {'command': 'command', 'timeout_millis': 30000}}
    with mock.patch('subprocess.run', return_value=subprocess.CompletedProcess(args=[], stdout=json.dumps(EXECUTABLE_SUCCESSFUL_OIDC_RESPONSE).encode('UTF-8'), returncode=0)):
        credentials = self.make_pluggable(credential_source=CREDENTIAL_SOURCE)
        subject_token = credentials.retrieve_subject_token(None)
        assert subject_token == self.EXECUTABLE_OIDC_TOKEN