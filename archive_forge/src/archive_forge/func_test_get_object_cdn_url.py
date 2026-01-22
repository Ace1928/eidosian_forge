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
def test_get_object_cdn_url(self):
    obj = self.driver.get_object(container_name='test_container200', object_name='test')
    url = urlparse.urlparse(self.driver.get_object_cdn_url(obj))
    query = urlparse.parse_qs(url.query)
    self.assertEqual(len(query['sig']), 1)
    self.assertGreater(len(query['sig'][0]), 0)