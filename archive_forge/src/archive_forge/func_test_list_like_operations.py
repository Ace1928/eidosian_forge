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
def test_list_like_operations(self):

    @base.VersionedObjectRegistry.register
    class MyElement(base.VersionedObject):
        fields = {'foo': fields.IntegerField()}

        def __init__(self, foo):
            super(MyElement, self).__init__()
            self.foo = foo

    class Foo(base.ObjectListBase, base.VersionedObject):
        fields = {'objects': fields.ListOfObjectsField('MyElement')}
    objlist = Foo(context='foo', objects=[MyElement(1), MyElement(2), MyElement(3)])
    self.assertEqual(list(objlist), objlist.objects)
    self.assertEqual(len(objlist), 3)
    self.assertIn(objlist.objects[0], objlist)
    self.assertEqual(list(objlist[:1]), [objlist.objects[0]])
    self.assertEqual(objlist[:1]._context, 'foo')
    self.assertEqual(objlist[2], objlist.objects[2])
    self.assertEqual(objlist.count(objlist.objects[0]), 1)
    self.assertEqual(objlist.index(objlist.objects[1]), 1)
    objlist.sort(key=lambda x: x.foo, reverse=True)
    self.assertEqual([3, 2, 1], [x.foo for x in objlist])