import copy
import datetime
import json
import os
import mock
import pytest  # type: ignore
import requests
import six
from google.auth import exceptions
from google.auth import jwt
import google.auth.transport.requests
from google.oauth2 import gdch_credentials
from google.oauth2.gdch_credentials import ServiceAccountCredentials
def test_refresh_wrong_requests_object(self):
    creds = ServiceAccountCredentials.from_service_account_info(self.INFO)
    creds = creds.with_gdch_audience(self.AUDIENCE)
    req = requests.Request()
    with pytest.raises(exceptions.RefreshError) as excinfo:
        creds.refresh(req)
    assert excinfo.match('request must be a google.auth.transport.requests.Request object')