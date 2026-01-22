import pytest
from google.auth import _helpers
from google.auth import exceptions
from google.auth import iam
from google.oauth2 import service_account
def test_refresh_no_scopes(http_request, credentials):
    with pytest.raises(exceptions.RefreshError):
        credentials.refresh(http_request)