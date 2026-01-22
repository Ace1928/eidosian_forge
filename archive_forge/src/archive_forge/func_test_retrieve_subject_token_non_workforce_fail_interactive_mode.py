import json
import os
import subprocess
import mock
import pytest  # type: ignore
from google.auth import exceptions
from google.auth import pluggable
from tests.test__default import WORKFORCE_AUDIENCE
@mock.patch.dict(os.environ, {'GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES': '1'})
def test_retrieve_subject_token_non_workforce_fail_interactive_mode(self):
    credentials = self.make_pluggable(credential_source=self.CREDENTIAL_SOURCE, interactive=True)
    with pytest.raises(ValueError) as excinfo:
        _ = credentials.retrieve_subject_token(None)
    assert excinfo.match('Interactive mode is only enabled for workforce pool.')