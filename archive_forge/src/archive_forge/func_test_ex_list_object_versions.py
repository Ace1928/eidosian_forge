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
def test_ex_list_object_versions(self):
    container = self.driver.list_containers()[0]
    container_id = container.extra['id']
    objects = self.driver.ex_list_object_versions(container_id=container_id)
    self.assertEqual(len(objects), 9)