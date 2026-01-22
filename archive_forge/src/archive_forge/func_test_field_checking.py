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
def test_field_checking(self):

    def create_class(field):

        @base.VersionedObjectRegistry.register
        class TestField(base.VersionedObject):
            VERSION = '1.5'
            fields = {'foo': field()}
        return TestField
    create_class(fields.DateTimeField)
    self.assertRaises(exception.ObjectFieldInvalid, create_class, fields.DateTime)
    self.assertRaises(exception.ObjectFieldInvalid, create_class, int)