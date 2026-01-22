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
def test_load_credentials_from_file_no_type(tmpdir):
    with pytest.raises(exceptions.DefaultCredentialsError) as excinfo:
        _default.load_credentials_from_file(test_default.CLIENT_SECRETS_FILE)
    assert excinfo.match('does not have a valid type')
    assert excinfo.match('Type is None')