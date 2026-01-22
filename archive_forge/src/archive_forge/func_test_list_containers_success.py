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
def test_list_containers_success(self):
    self.mock_response_klass.type = 'list_containers'
    AzureBlobsStorageDriver.RESPONSES_PER_REQUEST = 2
    containers = self.driver.list_containers()
    self.assertEqual(len(containers), 4)
    self.assertTrue('last_modified' in containers[1].extra)
    self.assertTrue('url' in containers[1].extra)
    self.assertTrue('etag' in containers[1].extra)
    self.assertTrue('lease' in containers[1].extra)
    self.assertTrue('meta_data' in containers[1].extra)
    self.assertEqual(containers[1].extra['etag'], '0x8CFBAB7B5B82D8E')