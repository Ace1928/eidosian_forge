import collections
import copy
import datetime
import hashlib
import inspect
from unittest import mock
import iso8601
from oslo_versionedobjects import base
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields
from oslo_versionedobjects import fixture
from oslo_versionedobjects import test
def test_object_registered_temporarily(self):
    self.assertRaises(exception.UnsupportedObjectError, FakeResource.obj_from_primitive, self.primitive)
    with fixture.VersionedObjectRegistryFixture() as obj_registry:
        obj_registry.setUp()
        obj_registry.register(FakeResource)
        obj = FakeResource.obj_from_primitive(self.primitive)
        self.assertEqual(obj.identifier, 123)
        self.assertEqual('1.0', obj.VERSION)
    self.assertRaises(exception.UnsupportedObjectError, FakeResource.obj_from_primitive, self.primitive)