import json
import os
import subprocess
import mock
import pytest  # type: ignore
from google.auth import exceptions
from google.auth import pluggable
from tests.test__default import WORKFORCE_AUDIENCE
@mock.patch.dict(os.environ, {'GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES': '1'})
def test_retrieve_subject_token_successfully(self, tmpdir):
    ACTUAL_CREDENTIAL_SOURCE_EXECUTABLE_OUTPUT_FILE = tmpdir.join('actual_output_file')
    ACTUAL_CREDENTIAL_SOURCE_EXECUTABLE = {'command': 'command', 'interactive_timeout_millis': 300000, 'output_file': ACTUAL_CREDENTIAL_SOURCE_EXECUTABLE_OUTPUT_FILE}
    ACTUAL_CREDENTIAL_SOURCE = {'executable': ACTUAL_CREDENTIAL_SOURCE_EXECUTABLE}
    testData = {'subject_token_oidc_id_token': {'stdout': json.dumps(self.EXECUTABLE_SUCCESSFUL_OIDC_RESPONSE_ID_TOKEN).encode('UTF-8'), 'impersonation_url': SERVICE_ACCOUNT_IMPERSONATION_URL, 'file_content': self.EXECUTABLE_SUCCESSFUL_OIDC_NO_EXPIRATION_TIME_RESPONSE_ID_TOKEN, 'expect_token': self.EXECUTABLE_OIDC_TOKEN}, 'subject_token_oidc_id_token_interacitve_mode': {'audience': WORKFORCE_AUDIENCE, 'file_content': self.EXECUTABLE_SUCCESSFUL_OIDC_NO_EXPIRATION_TIME_RESPONSE_ID_TOKEN, 'interactive': True, 'expect_token': self.EXECUTABLE_OIDC_TOKEN}, 'subject_token_oidc_jwt': {'stdout': json.dumps(self.EXECUTABLE_SUCCESSFUL_OIDC_RESPONSE_JWT).encode('UTF-8'), 'impersonation_url': SERVICE_ACCOUNT_IMPERSONATION_URL, 'file_content': self.EXECUTABLE_SUCCESSFUL_OIDC_NO_EXPIRATION_TIME_RESPONSE_JWT, 'expect_token': self.EXECUTABLE_OIDC_TOKEN}, 'subject_token_oidc_jwt_interactive_mode': {'audience': WORKFORCE_AUDIENCE, 'file_content': self.EXECUTABLE_SUCCESSFUL_OIDC_NO_EXPIRATION_TIME_RESPONSE_JWT, 'interactive': True, 'expect_token': self.EXECUTABLE_OIDC_TOKEN}, 'subject_token_saml': {'stdout': json.dumps(self.EXECUTABLE_SUCCESSFUL_SAML_RESPONSE).encode('UTF-8'), 'impersonation_url': SERVICE_ACCOUNT_IMPERSONATION_URL, 'file_content': self.EXECUTABLE_SUCCESSFUL_SAML_NO_EXPIRATION_TIME_RESPONSE, 'expect_token': self.EXECUTABLE_SAML_TOKEN}, 'subject_token_saml_interactive_mode': {'audience': WORKFORCE_AUDIENCE, 'file_content': self.EXECUTABLE_SUCCESSFUL_SAML_NO_EXPIRATION_TIME_RESPONSE, 'interactive': True, 'expect_token': self.EXECUTABLE_SAML_TOKEN}}
    for data in testData.values():
        with open(ACTUAL_CREDENTIAL_SOURCE_EXECUTABLE_OUTPUT_FILE, 'w') as output_file:
            json.dump(data.get('file_content'), output_file)
        with mock.patch('subprocess.run', return_value=subprocess.CompletedProcess(args=[], stdout=data.get('stdout'), returncode=0)):
            credentials = self.make_pluggable(audience=data.get('audience', AUDIENCE), service_account_impersonation_url=data.get('impersonation_url'), credential_source=ACTUAL_CREDENTIAL_SOURCE, interactive=data.get('interactive', False))
            subject_token = credentials.retrieve_subject_token(None)
            assert subject_token == data.get('expect_token')
        os.remove(ACTUAL_CREDENTIAL_SOURCE_EXECUTABLE_OUTPUT_FILE)