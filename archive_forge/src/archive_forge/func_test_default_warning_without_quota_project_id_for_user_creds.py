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
@mock.patch('google.auth._cloud_sdk.get_application_default_credentials_path', autospec=True)
def test_default_warning_without_quota_project_id_for_user_creds(get_adc_path):
    get_adc_path.return_value = test_default.AUTHORIZED_USER_CLOUD_SDK_FILE
    with pytest.warns(UserWarning, match='Cloud SDK'):
        credentials, project_id = _default.default_async(quota_project_id=None)