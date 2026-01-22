import datetime
import json
import os
import mock
import pytest  # type: ignore
from six.moves import http_client
from six.moves import urllib
from google.auth import _helpers
from google.auth import aws
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import transport
@mock.patch('google.auth._helpers.utcnow')
def test_retrieve_subject_token_success_environment_vars(self, utcnow, monkeypatch):
    monkeypatch.setenv(environment_vars.AWS_ACCESS_KEY_ID, ACCESS_KEY_ID)
    monkeypatch.setenv(environment_vars.AWS_SECRET_ACCESS_KEY, SECRET_ACCESS_KEY)
    monkeypatch.setenv(environment_vars.AWS_SESSION_TOKEN, TOKEN)
    monkeypatch.setenv(environment_vars.AWS_REGION, self.AWS_REGION)
    utcnow.return_value = datetime.datetime.strptime(self.AWS_SIGNATURE_TIME, '%Y-%m-%dT%H:%M:%SZ')
    credentials = self.make_credentials(credential_source=self.CREDENTIAL_SOURCE)
    subject_token = credentials.retrieve_subject_token(None)
    assert subject_token == self.make_serialized_aws_signed_request({'access_key_id': ACCESS_KEY_ID, 'secret_access_key': SECRET_ACCESS_KEY, 'security_token': TOKEN})