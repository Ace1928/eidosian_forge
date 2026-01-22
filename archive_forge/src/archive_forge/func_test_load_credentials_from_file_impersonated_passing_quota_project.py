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
def test_load_credentials_from_file_impersonated_passing_quota_project():
    credentials, _ = _default.load_credentials_from_file(IMPERSONATED_SERVICE_ACCOUNT_SERVICE_ACCOUNT_SOURCE_FILE, quota_project_id='new_quota_project')
    assert credentials._quota_project_id == 'new_quota_project'