import datetime
import os
import sys
import mock
import pytest  # type: ignore
from six.moves import reload_module
from google.auth import _oauth2client
@mock.patch('google.auth.app_engine.app_identity')
def test__convert_appengine_app_assertion_credentials(app_identity, mock_oauth2client_gae_imports):
    import oauth2client.contrib.appengine
    service_account_id = 'service_account_id'
    old_credentials = oauth2client.contrib.appengine.AppAssertionCredentials(scope='one two', service_account_id=service_account_id)
    new_credentials = _oauth2client._convert_appengine_app_assertion_credentials(old_credentials)
    assert new_credentials.scopes == ['one', 'two']
    assert new_credentials._service_account_id == old_credentials.service_account_id