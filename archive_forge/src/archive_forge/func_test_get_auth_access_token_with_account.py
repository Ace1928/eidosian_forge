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
def test_get_auth_access_token_with_account(check_output):
    check_output.return_value = b'access_token\n'
    token = _cloud_sdk.get_auth_access_token(account='account')
    assert token == 'access_token'
    check_output.assert_called_with(('gcloud', 'auth', 'print-access-token', '--account=account'), stderr=subprocess.STDOUT)