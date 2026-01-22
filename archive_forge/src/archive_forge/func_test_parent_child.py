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
def test_parent_child(self):

    @base.VersionedObjectRegistry.register
    class TestChild(base.VersionedObject):
        VERSION = '2.34'

    @base.VersionedObjectRegistry.register
    class TestObject(base.VersionedObject):
        VERSION = '1.23'
        fields = {'child': fields.ObjectField('TestChild')}
    tree = base.obj_tree_get_versions('TestObject')
    self.assertEqual({'TestObject': '1.23', 'TestChild': '2.34'}, tree)