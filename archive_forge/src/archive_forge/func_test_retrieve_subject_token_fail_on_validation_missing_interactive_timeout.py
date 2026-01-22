import json
import os
import subprocess
import mock
import pytest  # type: ignore
from google.auth import exceptions
from google.auth import pluggable
from tests.test__default import WORKFORCE_AUDIENCE
@mock.patch.dict(os.environ, {'GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES': '1'})
def test_retrieve_subject_token_fail_on_validation_missing_interactive_timeout(self):
    CREDENTIAL_SOURCE_EXECUTABLE = {'command': self.CREDENTIAL_SOURCE_EXECUTABLE_COMMAND, 'output_file': self.CREDENTIAL_SOURCE_EXECUTABLE_OUTPUT_FILE}
    CREDENTIAL_SOURCE = {'executable': CREDENTIAL_SOURCE_EXECUTABLE}
    credentials = self.make_pluggable(credential_source=CREDENTIAL_SOURCE, interactive=True)
    with pytest.raises(ValueError) as excinfo:
        _ = credentials.retrieve_subject_token(None)
    assert excinfo.match('Interactive mode cannot run without an interactive timeout.')