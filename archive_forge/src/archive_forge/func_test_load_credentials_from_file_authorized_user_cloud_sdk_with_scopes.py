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
def test_load_credentials_from_file_authorized_user_cloud_sdk_with_scopes():
    with pytest.warns(UserWarning, match='Cloud SDK'):
        credentials, project_id = _default.load_credentials_from_file(test_default.AUTHORIZED_USER_CLOUD_SDK_FILE, scopes=['https://www.google.com/calendar/feeds'])
    assert isinstance(credentials, google.oauth2._credentials_async.Credentials)
    assert project_id is None
    assert credentials.scopes == ['https://www.google.com/calendar/feeds']