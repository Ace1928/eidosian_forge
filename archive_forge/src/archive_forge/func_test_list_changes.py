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
def test_list_changes(self):

    @base.VersionedObjectRegistry.register
    class Foo(base.ObjectListBase, base.VersionedObject):
        fields = {'objects': fields.ListOfObjectsField('Bar')}

    @base.VersionedObjectRegistry.register
    class Bar(base.VersionedObject):
        fields = {'foo': fields.StringField()}
    obj = Foo(objects=[])
    self.assertEqual(set(['objects']), obj.obj_what_changed())
    obj.objects.append(Bar(foo='test'))
    self.assertEqual(set(['objects']), obj.obj_what_changed())
    obj.obj_reset_changes()
    self.assertEqual(set(['objects']), obj.obj_what_changed())
    obj.objects[0].obj_reset_changes()
    self.assertEqual(set(), obj.obj_what_changed())