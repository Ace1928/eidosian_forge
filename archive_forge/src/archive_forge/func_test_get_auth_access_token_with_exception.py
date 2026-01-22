import io
import json
import os
import subprocess
import sys
import mock
import pytest  # type: ignore
from google.auth import _cloud_sdk
from google.auth import environment_vars
from google.auth import exceptions
@mock.patch('subprocess.check_output', autospec=True)
def test_get_auth_access_token_with_exception(check_output):
    check_output.side_effect = OSError()
    with pytest.raises(exceptions.UserAccessTokenError):
        _cloud_sdk.get_auth_access_token(account='account')