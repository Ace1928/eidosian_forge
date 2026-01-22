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
def test_upload_big_block_object_success(self):
    _, file_path = tempfile.mkstemp(suffix='.jpg')
    file_size = AZURE_UPLOAD_CHUNK_SIZE + 1
    with open(file_path, 'w') as file_hdl:
        file_hdl.write('0' * file_size)
    container = Container(name='foo_bar_container', extra={}, driver=self.driver)
    object_name = 'foo_test_upload'
    extra = {'meta_data': {'some-value': 'foobar'}}
    obj = self.driver.upload_object(file_path=file_path, container=container, object_name=object_name, extra=extra, verify_hash=False)
    self.assertEqual(obj.name, 'foo_test_upload')
    self.assertEqual(obj.size, file_size)
    self.assertTrue('some-value' in obj.meta_data)
    os.remove(file_path)