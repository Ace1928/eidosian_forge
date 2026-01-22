import json
import os
import subprocess
import mock
import pytest  # type: ignore
from google.auth import exceptions
from google.auth import pluggable
from tests.test__default import WORKFORCE_AUDIENCE
@mock.patch.dict(os.environ, {'GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES': '1'})
def test_credential_source_timeout_small(self):
    with pytest.raises(ValueError) as excinfo:
        CREDENTIAL_SOURCE = {'executable': {'command': self.CREDENTIAL_SOURCE_EXECUTABLE_COMMAND, 'timeout_millis': 5000 - 1, 'output_file': self.CREDENTIAL_SOURCE_EXECUTABLE_OUTPUT_FILE}}
        _ = self.make_pluggable(credential_source=CREDENTIAL_SOURCE)
    assert excinfo.match('Timeout must be between 5 and 120 seconds.')