import os
import sys
import copy
import hmac
import math
import hashlib
import os.path  # pylint: disable-msg=W0404
from io import BytesIO
from hashlib import sha1
from unittest import mock
from unittest.mock import Mock, PropertyMock
import libcloud.utils.files
from libcloud.test import MockHttp  # pylint: disable-msg=E0611
from libcloud.test import unittest, make_response, generate_random_data
from libcloud.utils.py3 import StringIO, b, httplib, urlquote
from libcloud.utils.files import exhaust_iterator
from libcloud.common.types import MalformedResponseError
from libcloud.storage.base import CHUNK_SIZE, Object, Container
from libcloud.storage.types import (
from libcloud.test.storage.base import BaseRangeDownloadMockHttp
from libcloud.test.file_fixtures import StorageFileFixtures  # pylint: disable-msg=E0611
from libcloud.storage.drivers.cloudfiles import CloudFilesStorageDriver
def test_list_container_objects_iterator(self):
    CloudFilesMockHttp.type = 'ITERATOR'
    container = Container(name='test_container', extra={}, driver=self.driver)
    objects = self.driver.list_container_objects(container=container)
    self.assertEqual(len(objects), 5)
    obj = [o for o in objects if o.name == 'foo-test-1'][0]
    self.assertEqual(obj.hash, '16265549b5bda64ecdaa5156de4c97cc')
    self.assertEqual(obj.size, 1160520)
    self.assertEqual(obj.container.name, 'test_container')