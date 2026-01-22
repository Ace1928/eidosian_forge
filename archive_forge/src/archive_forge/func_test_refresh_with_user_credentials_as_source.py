import json
import pytest
import google.oauth2.credentials
from google.oauth2 import service_account
import google.auth.impersonated_credentials
from google.auth import _helpers
def test_refresh_with_user_credentials_as_source(authorized_user_file, impersonated_service_account_credentials, http_request, token_info):
    with open(authorized_user_file, 'r') as fh:
        info = json.load(fh)
    source_credentials = google.oauth2.credentials.Credentials(None, refresh_token=info['refresh_token'], token_uri=GOOGLE_OAUTH2_TOKEN_ENDPOINT, client_id=info['client_id'], client_secret=info['client_secret'], scopes=['https://www.googleapis.com/auth/cloud-platform'])
    source_credentials.refresh(http_request)
    target_scopes = ['https://www.googleapis.com/auth/devstorage.read_only', 'https://www.googleapis.com/auth/analytics']
    target_credentials = google.auth.impersonated_credentials.Credentials(source_credentials=source_credentials, target_principal=impersonated_service_account_credentials.service_account_email, target_scopes=target_scopes, lifetime=100)
    target_credentials.refresh(http_request)
    assert target_credentials.token