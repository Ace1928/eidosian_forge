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
def test_load_credentials_from_file_impersonated_with_quota_project():
    credentials, _ = _default.load_credentials_from_file(IMPERSONATED_SERVICE_ACCOUNT_WITH_QUOTA_PROJECT_FILE)
    assert isinstance(credentials, impersonated_credentials.Credentials)
    assert credentials._quota_project_id == 'quota_project'