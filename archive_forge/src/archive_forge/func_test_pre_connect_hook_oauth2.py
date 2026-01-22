import re
import sys
import copy
import json
import unittest
import email.utils
from io import BytesIO
from unittest import mock
from unittest.mock import Mock, PropertyMock
import pytest
from libcloud.test import StorageMockHttp
from libcloud.utils.py3 import StringIO, httplib
from libcloud.common.types import InvalidCredsError
from libcloud.storage.base import Object, Container
from libcloud.test.secrets import STORAGE_GOOGLE_STORAGE_PARAMS
from libcloud.common.google import GoogleAuthType
from libcloud.storage.drivers import google_storage
from libcloud.test.file_fixtures import StorageFileFixtures
from libcloud.test.storage.test_s3 import S3Tests, S3MockHttp
from libcloud.test.common.test_google import GoogleTestCase
@mock.patch('libcloud.common.google.GoogleOAuth2Credential')
def test_pre_connect_hook_oauth2(self, mock_oauth2_credential_init):
    mock_oauth2_credential_init.return_value = mock.Mock()
    starting_params = {'starting': 'params'}
    starting_headers = {'starting': 'headers'}
    conn = CONN_CLS('foo_user', 'bar_key', secure=True, auth_type=GoogleAuthType.GCE)
    conn._get_s3_auth_signature = mock.Mock()
    conn.oauth2_credential = mock.Mock()
    conn.oauth2_credential.access_token = 'Access_Token!'
    expected_headers = dict(starting_headers)
    expected_headers['Authorization'] = 'Bearer Access_Token!'
    result = conn.pre_connect_hook(dict(starting_params), dict(starting_headers))
    self.assertEqual(result, (starting_params, expected_headers))