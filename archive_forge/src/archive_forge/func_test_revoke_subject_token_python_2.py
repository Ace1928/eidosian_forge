import json
import os
import subprocess
import mock
import pytest  # type: ignore
from google.auth import exceptions
from google.auth import pluggable
from tests.test__default import WORKFORCE_AUDIENCE
@mock.patch.dict(os.environ, {'GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES': '1'})
def test_revoke_subject_token_python_2(self):
    with mock.patch('sys.version_info', (2, 7)):
        credentials = self.make_pluggable(audience=WORKFORCE_AUDIENCE, credential_source=self.CREDENTIAL_SOURCE, interactive=True)
        with pytest.raises(exceptions.RefreshError) as excinfo:
            _ = credentials.revoke(None)
        assert excinfo.match('Pluggable auth is only supported for python 3.6+')