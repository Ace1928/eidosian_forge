import json
import os
import subprocess
import mock
import pytest  # type: ignore
from google.auth import exceptions
from google.auth import pluggable
from tests.test__default import WORKFORCE_AUDIENCE
@mock.patch.dict(os.environ, {'GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES': '1'})
def test_retrieve_subject_token_file_cache_refresh_error_retry(self, tmpdir):
    ACTUAL_CREDENTIAL_SOURCE_EXECUTABLE_OUTPUT_FILE = tmpdir.join('actual_output_file')
    ACTUAL_CREDENTIAL_SOURCE_EXECUTABLE = {'command': 'command', 'timeout_millis': 30000, 'output_file': ACTUAL_CREDENTIAL_SOURCE_EXECUTABLE_OUTPUT_FILE}
    ACTUAL_CREDENTIAL_SOURCE = {'executable': ACTUAL_CREDENTIAL_SOURCE_EXECUTABLE}
    ACTUAL_EXECUTABLE_RESPONSE = {'version': 2, 'success': True, 'token_type': 'urn:ietf:params:oauth:token-type:id_token', 'id_token': self.EXECUTABLE_OIDC_TOKEN, 'expiration_time': 9999999999}
    with open(ACTUAL_CREDENTIAL_SOURCE_EXECUTABLE_OUTPUT_FILE, 'w') as output_file:
        json.dump(ACTUAL_EXECUTABLE_RESPONSE, output_file)
    with mock.patch('subprocess.run', return_value=subprocess.CompletedProcess(args=[], stdout=json.dumps(self.EXECUTABLE_SUCCESSFUL_OIDC_RESPONSE_ID_TOKEN).encode('UTF-8'), returncode=0)):
        credentials = self.make_pluggable(credential_source=ACTUAL_CREDENTIAL_SOURCE)
        subject_token = credentials.retrieve_subject_token(None)
        assert subject_token == self.EXECUTABLE_OIDC_TOKEN
    os.remove(ACTUAL_CREDENTIAL_SOURCE_EXECUTABLE_OUTPUT_FILE)