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
def test_upload_blob_object_via_stream_from_iterable(self):
    self.mock_response_klass.use_param = 'comp'
    container = Container(name='foo_bar_container', extra={}, driver=self.driver)
    object_name = 'foo_test_upload'
    iterator = iter([b('34'), b('5')])
    extra = {'content_type': 'text/plain'}
    obj = self.driver.upload_object_via_stream(container=container, object_name=object_name, iterator=iterator, extra=extra)
    self.assertEqual(obj.name, object_name)
    self.assertEqual(obj.size, 3)
    self.mock_response_klass.use_param = None