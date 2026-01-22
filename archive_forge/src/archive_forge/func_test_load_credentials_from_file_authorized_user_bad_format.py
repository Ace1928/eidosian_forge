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
def test_load_credentials_from_file_authorized_user_bad_format(tmpdir):
    filename = tmpdir.join('authorized_user_bad.json')
    filename.write(json.dumps({'type': 'authorized_user'}))
    with pytest.raises(exceptions.DefaultCredentialsError) as excinfo:
        _default.load_credentials_from_file(str(filename))
    assert excinfo.match('Failed to load authorized user')
    assert excinfo.match('missing fields')