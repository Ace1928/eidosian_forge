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
def test_set_permissions(self):
    mock_request = mock.Mock()
    self.driver.json_connection.request = mock_request
    self.driver.ex_set_permissions('bucket', 'object', entity='user-foo', role='OWNER')
    url = '/storage/v1/b/bucket/o/object/acl'
    mock_request.assert_called_once_with(url, method='POST', data=json.dumps({'role': 'OWNER', 'entity': 'user-foo'}))
    mock_request.reset_mock()
    self.driver.ex_set_permissions('bucket', 'object', entity='user-foo', role=google_storage.ObjectPermissions.OWNER)
    url = '/storage/v1/b/bucket/o/object/acl'
    mock_request.assert_called_once_with(url, method='POST', data=json.dumps({'role': 'OWNER', 'entity': 'user-foo'}))
    mock_request.reset_mock()
    self.driver.ex_set_permissions('bucket', entity='user-foo', role='OWNER')
    url = '/storage/v1/b/bucket/acl'
    mock_request.assert_called_once_with(url, method='POST', data=json.dumps({'role': 'OWNER', 'entity': 'user-foo'}))
    mock_request.reset_mock()
    self.driver.ex_set_permissions('bucket', entity='user-foo', role=google_storage.ContainerPermissions.OWNER)
    url = '/storage/v1/b/bucket/acl'
    mock_request.assert_called_once_with(url, method='POST', data=json.dumps({'role': 'OWNER', 'entity': 'user-foo'}))