import datetime
import json
import os
import pickle
import sys
import mock
import pytest  # type: ignore
from google.auth import _helpers
from google.auth import exceptions
from google.oauth2 import _credentials_async as _credentials_async
from google.oauth2 import credentials
from tests.oauth2 import test_credentials
def test_with_quota_project(self):
    cred = _credentials_async.UserAccessTokenCredentials()
    quota_project_cred = cred.with_quota_project('project-foo')
    assert quota_project_cred._quota_project_id == 'project-foo'
    assert quota_project_cred._account == cred._account