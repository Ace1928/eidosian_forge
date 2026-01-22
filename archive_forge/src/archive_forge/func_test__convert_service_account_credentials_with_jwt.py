import datetime
import os
import sys
import mock
import pytest  # type: ignore
from six.moves import reload_module
from google.auth import _oauth2client
def test__convert_service_account_credentials_with_jwt():
    old_class = oauth2client.service_account._JWTAccessCredentials
    old_credentials = old_class.from_json_keyfile_name(SERVICE_ACCOUNT_JSON_FILE)
    new_credentials = _oauth2client._convert_service_account_credentials(old_credentials)
    assert new_credentials.service_account_email == old_credentials.service_account_email
    assert new_credentials._signer.key_id == old_credentials._private_key_id
    assert new_credentials._token_uri == old_credentials.token_uri