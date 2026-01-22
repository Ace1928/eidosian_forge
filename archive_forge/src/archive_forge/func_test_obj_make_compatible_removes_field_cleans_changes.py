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
def test_obj_make_compatible_removes_field_cleans_changes(self):

    @base.VersionedObjectRegistry.register_if(False)
    class TestObject(base.VersionedObject):
        VERSION = '1.1'
        fields = {'foo': fields.StringField(), 'bar': fields.StringField()}

        def obj_make_compatible(self, primitive, target_version):
            del primitive['bar']
    obj = TestObject(foo='test1', bar='test2')
    prim = obj.obj_to_primitive('1.0')
    self.assertEqual(['foo'], prim['versioned_object.changes'])