import os
import sys
import hmac
import base64
import tempfile
from io import BytesIO
from hashlib import sha1
from unittest import mock
from unittest.mock import Mock, PropertyMock
import libcloud.utils.files  # NOQA: F401
from libcloud.test import MockHttp  # pylint: disable-msg=E0611  # noqa
from libcloud.test import unittest, make_response, generate_random_data
from libcloud.utils.py3 import ET, StringIO, b, httplib, parse_qs, urlparse
from libcloud.utils.files import exhaust_iterator
from libcloud.common.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.storage.base import Object, Container
from libcloud.test.secrets import STORAGE_S3_PARAMS
from libcloud.storage.types import (
from libcloud.test.storage.base import BaseRangeDownloadMockHttp
from libcloud.storage.drivers.s3 import (
from libcloud.test.file_fixtures import StorageFileFixtures  # pylint: disable-msg=E0611
def test_region_keyword_argument(self):
    driver = S3StorageDriver(*self.driver_args)
    self.assertEqual(driver.region, 'us-east-1')
    self.assertEqual(driver.connection.host, 's3.amazonaws.com')
    driver = S3StorageDriver(*self.driver_args, region='us-west-2')
    self.assertEqual(driver.region, 'us-west-2')
    self.assertEqual(driver.connection.host, 's3-us-west-2.amazonaws.com')
    driver1 = S3StorageDriver(*self.driver_args, region='us-west-2')
    self.assertEqual(driver1.region, 'us-west-2')
    self.assertEqual(driver1.connection.host, 's3-us-west-2.amazonaws.com')
    driver2 = S3StorageDriver(*self.driver_args, region='ap-south-1')
    self.assertEqual(driver2.region, 'ap-south-1')
    self.assertEqual(driver2.connection.host, 's3-ap-south-1.amazonaws.com')
    self.assertEqual(driver1.region, 'us-west-2')
    self.assertEqual(driver1.connection.host, 's3-us-west-2.amazonaws.com')
    for region in S3StorageDriver.list_regions():
        driver = S3StorageDriver(*self.driver_args, region=region)
        self.assertEqual(driver.region, region)
    expected_msg = 'Invalid or unsupported region: foo'
    self.assertRaisesRegex(ValueError, expected_msg, S3StorageDriver, *self.driver_args, region='foo')
    driver3 = S3StorageDriver(*self.driver_args, region='ap-south-1', host='host1.bar.com')
    self.assertEqual(driver3.region, 'ap-south-1')
    self.assertEqual(driver3.connection.host, 'host1.bar.com')
    driver4 = S3StorageDriver(*self.driver_args, host='host2.bar.com')
    self.assertEqual(driver4.connection.host, 'host2.bar.com')