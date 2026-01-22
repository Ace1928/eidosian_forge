import os
import sys
import time
import shutil
import platform
import tempfile
import unittest
import multiprocessing
from libcloud.utils.files import exhaust_iterator
from libcloud.common.types import LibcloudError
from libcloud.storage.base import Object, Container
from libcloud.storage.types import (
def test_objects_success(self):
    tmppath = self.make_tmp_file()
    container = self.driver.create_container('test3')
    obj1 = container.upload_object(tmppath, 'object1')
    obj2 = container.upload_object(tmppath, 'path/object2')
    obj3 = container.upload_object(tmppath, 'path/to/object3')
    obj4 = container.upload_object(tmppath, 'path/to/object4.ext')
    with open(tmppath, 'rb') as tmpfile:
        obj5 = container.upload_object_via_stream(tmpfile, 'object5')
    obj6 = container.upload_object(tmppath, 'foo5')
    obj7 = container.upload_object(tmppath, 'foo6')
    obj8 = container.upload_object(tmppath, 'Afoo7')
    objects = self.driver.list_container_objects(container=container)
    self.assertEqual(len(objects), 8)
    prefix = os.path.join('path', 'invalid')
    objects = self.driver.list_container_objects(container=container, prefix=prefix)
    self.assertEqual(len(objects), 0)
    prefix = os.path.join('path', 'to')
    objects = self.driver.list_container_objects(container=container, prefix=prefix)
    self.assertEqual(len(objects), 2)
    self.assertEqual(objects[0].name, 'path/to/object3')
    self.assertEqual(objects[1].name, 'path/to/object4.ext')
    prefix = 'foo'
    objects = self.driver.list_container_objects(container=container, prefix=prefix)
    self.assertEqual(len(objects), 2)
    self.assertEqual(objects[0].name, 'foo5')
    self.assertEqual(objects[1].name, 'foo6')
    prefix = 'foo5'
    objects = self.driver.list_container_objects(container=container, prefix=prefix)
    self.assertEqual(len(objects), 1)
    self.assertEqual(objects[0].name, 'foo5')
    prefix = 'foo6'
    objects = self.driver.list_container_objects(container=container, prefix=prefix)
    self.assertEqual(len(objects), 1)
    self.assertEqual(objects[0].name, 'foo6')
    for obj in objects:
        self.assertNotEqual(obj.hash, None)
        self.assertEqual(obj.size, 4096)
        self.assertEqual(obj.container.name, 'test3')
        self.assertTrue('creation_time' in obj.extra)
        self.assertTrue('modify_time' in obj.extra)
        self.assertTrue('access_time' in obj.extra)
    obj1.delete()
    obj2.delete()
    objects = container.list_objects()
    self.assertEqual(len(objects), 6)
    container.delete_object(obj3)
    container.delete_object(obj4)
    container.delete_object(obj5)
    container.delete_object(obj6)
    container.delete_object(obj7)
    container.delete_object(obj8)
    objects = container.list_objects()
    self.assertEqual(len(objects), 0)
    container.delete()
    self.remove_tmp_file(tmppath)