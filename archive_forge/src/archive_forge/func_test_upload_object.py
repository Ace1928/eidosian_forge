import os
import sys
import json
import tempfile
from unittest import mock
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import b, httplib
from libcloud.utils.files import exhaust_iterator
from libcloud.test.file_fixtures import StorageFileFixtures
from libcloud.storage.drivers.backblaze_b2 import BackblazeB2StorageDriver
def test_upload_object(self):
    file_path = os.path.abspath(__file__)
    container = self.driver.list_containers()[0]
    obj = self.driver.upload_object(file_path=file_path, container=container, object_name='test0007.txt')
    self.assertEqual(obj.name, 'test0007.txt')
    self.assertEqual(obj.size, 24)
    self.assertEqual(obj.extra['fileId'], 'abcde')