import json
import os
import subprocess
import mock
import pytest  # type: ignore
from google.auth import exceptions
from google.auth import pluggable
from tests.test__default import WORKFORCE_AUDIENCE
@mock.patch.dict(os.environ, {'GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES': '1'})
def test_credential_source_missing_output_interactive_mode(self):
    CREDENTIAL_SOURCE = {'executable': {'command': self.CREDENTIAL_SOURCE_EXECUTABLE_COMMAND}}
    credentials = self.make_pluggable(credential_source=CREDENTIAL_SOURCE, interactive=True)
    with pytest.raises(ValueError) as excinfo:
        _ = credentials.retrieve_subject_token(None)
    assert excinfo.match('An output_file must be specified in the credential configuration for interactive mode.')