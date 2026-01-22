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
def test_set_permissions_no_entity(self):
    mock_request = mock.Mock()
    mock_get_user = mock.Mock(return_value=None)
    self.driver._get_user = mock_get_user
    self.driver.json_connection.request = mock_request
    self.assertRaises(ValueError, self.driver.ex_set_permissions, 'bucket', 'object', role='OWNER')
    self.assertRaises(ValueError, self.driver.ex_set_permissions, 'bucket', role='OWNER')
    mock_request.assert_not_called()
    mock_get_user.return_value = 'foo@foo.com'
    self.driver.ex_set_permissions('bucket', 'object', role='OWNER')
    url = '/storage/v1/b/bucket/o/object/acl'
    mock_request.assert_called_once_with(url, method='POST', data=json.dumps({'role': 'OWNER', 'entity': 'user-foo@foo.com'}))
    mock_request.reset_mock()
    mock_get_user.return_value = 'foo@foo.com'
    self.driver.ex_set_permissions('bucket', role='OWNER')
    url = '/storage/v1/b/bucket/acl'
    mock_request.assert_called_once_with(url, method='POST', data=json.dumps({'role': 'OWNER', 'entity': 'user-foo@foo.com'}))