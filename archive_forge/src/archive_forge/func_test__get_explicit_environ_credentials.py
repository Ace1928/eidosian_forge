import json
import os
import mock
import pytest  # type: ignore
from google.auth import _credentials_async as credentials
from google.auth import _default_async as _default
from google.auth import app_engine
from google.auth import compute_engine
from google.auth import environment_vars
from google.auth import exceptions
from google.oauth2 import _service_account_async as service_account
import google.oauth2.credentials
from tests import test__default as test_default
@pytest.mark.parametrize('quota_project_id', [None, 'project-foo'])
@LOAD_FILE_PATCH
def test__get_explicit_environ_credentials(load, quota_project_id, monkeypatch):
    monkeypatch.setenv(environment_vars.CREDENTIALS, 'filename')
    credentials, project_id = _default._get_explicit_environ_credentials(quota_project_id=quota_project_id)
    assert credentials is MOCK_CREDENTIALS
    assert project_id is mock.sentinel.project_id
    load.assert_called_with('filename', quota_project_id=quota_project_id)