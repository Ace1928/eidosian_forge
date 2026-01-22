import os
import sys
import unittest
from unittest import mock
from libcloud.test import MockHttp  # pylint: disable-msg=E0611
from libcloud.test import make_response, generate_random_data
from libcloud.utils.py3 import b, httplib, parse_qs, urlparse
from libcloud.common.types import InvalidCredsError
from libcloud.storage.base import Object, Container
from libcloud.test.secrets import STORAGE_OSS_PARAMS
from libcloud.storage.types import (
from libcloud.test.file_fixtures import StorageFileFixtures  # pylint: disable-msg=E0611
from libcloud.storage.drivers.oss import CHUNK_SIZE, OSSConnection, OSSStorageDriver
from libcloud.storage.drivers.dummy import DummyIterator
def test_create_container_with_ex_location(self):
    self.mock_response_klass.type = 'create_container_location'
    name = 'new_container'
    self.ex_location = 'oss-cn-beijing'
    container = self.driver.create_container(container_name=name, ex_location=self.ex_location)
    self.assertEqual(container.name, name)
    self.assertTrue(container.extra['location'], self.ex_location)