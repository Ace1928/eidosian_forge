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
def test_pre_connect_hook_hmac(self):
    starting_params = {'starting': 'params'}
    starting_headers = {'starting': 'headers'}

    def fake_hmac_method(params, headers):
        fake_hmac_method.params_passed = copy.deepcopy(params)
        fake_hmac_method.headers_passed = copy.deepcopy(headers)
        return 'fake signature!'
    conn = CONN_CLS('foo_user', 'bar_key', secure=True, auth_type=GoogleAuthType.GCS_S3)
    conn._get_s3_auth_signature = fake_hmac_method
    conn.action = 'GET'
    conn.method = '/foo'
    expected_headers = dict(starting_headers)
    expected_headers['Authorization'] = '{} {}:{}'.format(google_storage.SIGNATURE_IDENTIFIER, 'foo_user', 'fake signature!')
    result = conn.pre_connect_hook(dict(starting_params), dict(starting_headers))
    self.assertEqual(result, (dict(starting_params), expected_headers))
    self.assertEqual(fake_hmac_method.params_passed, starting_params)
    self.assertEqual(fake_hmac_method.headers_passed, starting_headers)
    self.assertIsNone(conn.oauth2_credential)