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
@mock.patch('oslo_versionedobjects.base.VersionedObject.indirection_api')
def test_serializer_calls_old_backport_interface(self, indirection_api):

    @base.VersionedObjectRegistry.register
    class MyOldObj(base.VersionedObject):
        pass
    ser = base.VersionedObjectSerializer()
    prim = MyOldObj(foo=1).obj_to_primitive()
    prim['versioned_object.version'] = '2.0'
    indirection_api.object_backport_versions.side_effect = NotImplementedError('Old')
    ser.deserialize_entity(mock.sentinel.context, prim)
    indirection_api.object_backport.assert_called_once_with(mock.sentinel.context, prim, '1.0')