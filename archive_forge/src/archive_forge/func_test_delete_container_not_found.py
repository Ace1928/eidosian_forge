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
def test_delete_container_not_found(self):
    self.mock_response_klass.type = 'NOT_FOUND'
    container = Container(name='foo_bar_container', extra={}, driver=self.driver)
    try:
        self.driver.delete_container(container=container)
    except ContainerDoesNotExistError:
        pass
    else:
        self.fail('Container does not exist but an exception was not' + 'thrown')