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
def test_obj_make_compatible_child_versions(self):

    @base.VersionedObjectRegistry.register
    class MyElement(base.VersionedObject):
        fields = {'foo': fields.IntegerField()}

    @base.VersionedObjectRegistry.register
    class Foo(base.ObjectListBase, base.VersionedObject):
        VERSION = '1.1'
        fields = {'objects': fields.ListOfObjectsField('MyElement')}
        child_versions = {'1.0': '1.0', '1.1': '1.0'}
    subobj = MyElement(foo=1)
    obj = Foo(objects=[subobj])
    primitive = obj.obj_to_primitive()['versioned_object.data']
    with mock.patch.object(subobj, 'obj_make_compatible') as mock_compat:
        obj.obj_make_compatible(copy.copy(primitive), '1.1')
        self.assertTrue(mock_compat.called)