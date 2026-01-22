import sys
import base64
import os.path
import unittest
import libcloud.utils.files
from libcloud.test import MockHttp, make_response, generate_random_data
from libcloud.utils.py3 import b, httplib, urlparse
from libcloud.common.types import LibcloudError
from libcloud.storage.base import Object, Container
from libcloud.storage.types import (
from libcloud.test.file_fixtures import StorageFileFixtures
from libcloud.storage.drivers.atmos import AtmosDriver, AtmosConnection
from libcloud.storage.drivers.dummy import DummyIterator
def test_download_object_success_not_found(self):
    AtmosMockHttp.type = 'NOT_FOUND'
    container = Container(name='foo_bar_container', extra={}, driver=self.driver)
    obj = Object(name='foo_bar_object', size=1000, hash=None, extra={}, container=container, meta_data=None, driver=self.driver)
    destination_path = os.path.abspath(__file__) + '.temp'
    try:
        self.driver.download_object(obj=obj, destination_path=destination_path, overwrite_existing=False, delete_on_failure=True)
    except ObjectDoesNotExistError:
        pass
    else:
        self.fail('Object does not exist but an exception was not thrown')