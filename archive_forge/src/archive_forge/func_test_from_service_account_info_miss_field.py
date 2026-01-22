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
def test_from_service_account_info_miss_field(self):
    for field in ['format_version', 'private_key_id', 'private_key', 'name', 'project', 'token_uri']:
        info_with_missing_field = copy.deepcopy(self.INFO)
        del info_with_missing_field[field]
        with pytest.raises(ValueError) as excinfo:
            ServiceAccountCredentials.from_service_account_info(info_with_missing_field)
        assert excinfo.match('missing fields')