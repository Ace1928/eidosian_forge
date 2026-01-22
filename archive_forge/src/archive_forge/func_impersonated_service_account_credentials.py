import json
import pytest
import google.oauth2.credentials
from google.oauth2 import service_account
import google.auth.impersonated_credentials
from google.auth import _helpers
@pytest.fixture
def impersonated_service_account_credentials(impersonated_service_account_file):
    yield service_account.Credentials.from_service_account_file(impersonated_service_account_file)