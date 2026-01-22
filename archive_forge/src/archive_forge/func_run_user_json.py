import json
import os
from six.moves import http_client
import oauth2client
from oauth2client import client
from oauth2client import service_account
from oauth2client import transport
def run_user_json():
    with open(USER_KEY_PATH, 'r') as file_object:
        client_credentials = json.load(file_object)
    credentials = client.GoogleCredentials(access_token=None, client_id=client_credentials['client_id'], client_secret=client_credentials['client_secret'], refresh_token=client_credentials['refresh_token'], token_expiry=None, token_uri=oauth2client.GOOGLE_TOKEN_URI, user_agent='Python client library')
    _check_user_info(credentials, USER_KEY_EMAIL)