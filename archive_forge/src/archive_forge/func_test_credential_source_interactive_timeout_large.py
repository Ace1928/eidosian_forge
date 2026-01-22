import json
import os
import subprocess
import mock
import pytest  # type: ignore
from google.auth import exceptions
from google.auth import pluggable
from tests.test__default import WORKFORCE_AUDIENCE
@mock.patch.dict(os.environ, {'GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES': '1'})
def test_credential_source_interactive_timeout_large(self):
    with pytest.raises(ValueError) as excinfo:
        CREDENTIAL_SOURCE = {'executable': {'command': self.CREDENTIAL_SOURCE_EXECUTABLE_COMMAND, 'interactive_timeout_millis': 1800000 + 1, 'output_file': self.CREDENTIAL_SOURCE_EXECUTABLE_OUTPUT_FILE}}
        _ = self.make_pluggable(credential_source=CREDENTIAL_SOURCE)
    assert excinfo.match('Interactive timeout must be between 30 seconds and 30 minutes.')