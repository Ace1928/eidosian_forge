import os
import sys
import json
import tempfile
from io import BytesIO
from libcloud.test import generate_random_data  # pylint: disable-msg=E0611
from libcloud.test import unittest
from libcloud.utils.py3 import b, httplib, parse_qs, urlparse, basestring
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.storage.base import Object, Container
from libcloud.test.secrets import STORAGE_AZURE_BLOBS_PARAMS, STORAGE_AZURITE_BLOBS_PARAMS
from libcloud.storage.types import (
from libcloud.test.storage.base import BaseRangeDownloadMockHttp
from libcloud.test.file_fixtures import StorageFileFixtures  # pylint: disable-msg=E0611
from libcloud.storage.drivers.azure_blobs import (
def test_storage_driver_host(self):
    driver1 = self.driver_type('fakeaccount1', 'deadbeafcafebabe==')
    driver2 = self.driver_type('fakeaccount2', 'deadbeafcafebabe==')
    driver3 = self.driver_type('fakeaccount3', 'deadbeafcafebabe==', host='test.foo.bar.com')
    host1 = driver1.connection.host
    host2 = driver2.connection.host
    host3 = driver3.connection.host
    self.assertEqual(host1, 'fakeaccount1.blob.core.windows.net')
    self.assertEqual(host2, 'fakeaccount2.blob.core.windows.net')
    self.assertEqual(host3, 'test.foo.bar.com')