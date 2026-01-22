import json
import os
import mock
import pytest  # type: ignore
from google.auth import _default
from google.auth import api_key
from google.auth import app_engine
from google.auth import aws
from google.auth import compute_engine
from google.auth import credentials
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import external_account
from google.auth import external_account_authorized_user
from google.auth import identity_pool
from google.auth import impersonated_credentials
from google.auth import pluggable
from google.oauth2 import gdch_credentials
from google.oauth2 import service_account
import google.oauth2.credentials
@mock.patch('google.auth._default._get_explicit_environ_credentials', return_value=(MOCK_CREDENTIALS, mock.sentinel.project_id), autospec=True)
def test_default_quota_project(with_quota_project):
    credentials, project_id = _default.default(quota_project_id='project-foo')
    MOCK_CREDENTIALS.with_quota_project.assert_called_once_with('project-foo')
    assert project_id == mock.sentinel.project_id