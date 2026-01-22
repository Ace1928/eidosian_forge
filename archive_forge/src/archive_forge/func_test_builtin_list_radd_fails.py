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
def test_builtin_list_radd_fails(self):

    @base.VersionedObjectRegistry.register_if(False)
    class MyList(base.ObjectListBase, base.VersionedObject):
        fields = {'objects': fields.ListOfObjectsField('MyOwnedObject')}
    list1 = MyList(objects=[MyOwnedObject(baz=1)])

    def add(obj):
        return [] + obj
    self.assertRaises(TypeError, add, list1)