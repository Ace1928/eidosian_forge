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
def test__get_gcloud_sdk_credentials_non_existent(get_adc_path, tmpdir):
    non_existent = tmpdir.join('non-existent')
    get_adc_path.return_value = str(non_existent)
    credentials, project_id = _default._get_gcloud_sdk_credentials()
    assert credentials is None
    assert project_id is None