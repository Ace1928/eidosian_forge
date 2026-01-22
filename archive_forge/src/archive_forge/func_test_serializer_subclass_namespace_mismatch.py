import copy
import datetime
import jsonschema
import logging
import pytz
from unittest import mock
from oslo_context import context
from oslo_serialization import jsonutils
from oslo_utils import timeutils
import testtools
from testtools import matchers
from oslo_versionedobjects import base
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields
from oslo_versionedobjects import fixture
from oslo_versionedobjects import test
def test_serializer_subclass_namespace_mismatch(self):

    @base.VersionedObjectRegistry.register
    class MyNSObj(base.VersionedObject):
        OBJ_SERIAL_NAMESPACE = 'foo'
        fields = {'foo': fields.IntegerField()}

    class MySerializer(base.VersionedObjectSerializer):
        OBJ_BASE_CLASS = MyNSObj
    myser = MySerializer()
    voser = base.VersionedObjectSerializer()
    obj = MyObj(foo=123)
    obj2 = myser.deserialize_entity(None, voser.serialize_entity(None, obj))
    self.assertNotIsInstance(obj2, MyNSObj)
    self.assertIn('versioned_object.name', obj2)