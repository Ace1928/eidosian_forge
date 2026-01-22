import json
import os.path
import shutil
import tempfile
import unittest
import mock
import six
from apitools.base.py import credentials_lib
from apitools.base.py import util
def testGceAssertionCredentialsToJson(self):
    scopes = ['scope1']
    service_account_name = 'my_service_account'
    original_creds = self._GetServiceCreds(service_account_name=service_account_name, scopes=scopes)
    original_creds_json_str = original_creds.to_json()
    json.loads(original_creds_json_str)