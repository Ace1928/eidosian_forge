import json
import os
import subprocess
import mock
import pytest  # type: ignore
from google.auth import exceptions
from google.auth import pluggable
from tests.test__default import WORKFORCE_AUDIENCE
@mock.patch.dict(os.environ, {'GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES': '1'})
def test_retrieve_subject_token_file_cache_value_error_report(self, tmpdir):
    ACTUAL_CREDENTIAL_SOURCE_EXECUTABLE_OUTPUT_FILE = tmpdir.join('actual_output_file')
    ACTUAL_CREDENTIAL_SOURCE_EXECUTABLE = {'command': 'command', 'timeout_millis': 30000, 'output_file': ACTUAL_CREDENTIAL_SOURCE_EXECUTABLE_OUTPUT_FILE}
    ACTUAL_CREDENTIAL_SOURCE = {'executable': ACTUAL_CREDENTIAL_SOURCE_EXECUTABLE}
    ACTUAL_EXECUTABLE_RESPONSE = {'success': True, 'token_type': 'urn:ietf:params:oauth:token-type:id_token', 'id_token': self.EXECUTABLE_OIDC_TOKEN, 'expiration_time': 9999999999}
    with open(ACTUAL_CREDENTIAL_SOURCE_EXECUTABLE_OUTPUT_FILE, 'w') as output_file:
        json.dump(ACTUAL_EXECUTABLE_RESPONSE, output_file)
    credentials = self.make_pluggable(credential_source=ACTUAL_CREDENTIAL_SOURCE)
    with pytest.raises(ValueError) as excinfo:
        _ = credentials.retrieve_subject_token(None)
    assert excinfo.match('The executable response is missing the version field.')
    os.remove(ACTUAL_CREDENTIAL_SOURCE_EXECUTABLE_OUTPUT_FILE)