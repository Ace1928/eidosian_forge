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
def test__get_gae_credentials_env_unset():
    assert environment_vars.LEGACY_APPENGINE_RUNTIME not in os.environ
    assert 'GAE_RUNTIME' not in os.environ
    credentials, project_id = _default._get_gae_credentials()
    assert credentials is None
    assert project_id is None