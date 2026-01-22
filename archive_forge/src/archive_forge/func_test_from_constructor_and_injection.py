import json
import os
import subprocess
import mock
import pytest  # type: ignore
from google.auth import exceptions
from google.auth import pluggable
from tests.test__default import WORKFORCE_AUDIENCE
def test_from_constructor_and_injection(self):
    credentials = pluggable.Credentials(audience=AUDIENCE, subject_token_type=SUBJECT_TOKEN_TYPE, token_url=TOKEN_URL, token_info_url=TOKEN_INFO_URL, credential_source=self.CREDENTIAL_SOURCE, interactive=True)
    setattr(credentials, '_tokeninfo_username', 'mock_external_account_id')
    assert isinstance(credentials, pluggable.Credentials)
    assert credentials.interactive
    assert credentials.external_account_id == 'mock_external_account_id'