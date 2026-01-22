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
def test_loaded_in_primitive(self):
    obj = MyObj(foo=1)
    obj.obj_reset_changes()
    self.assertEqual(obj.bar, 'loaded!')
    expected = {'versioned_object.name': 'MyObj', 'versioned_object.namespace': 'versionedobjects', 'versioned_object.version': '1.6', 'versioned_object.changes': ['bar'], 'versioned_object.data': {'foo': 1, 'bar': 'loaded!'}}
    self.assertEqual(obj.obj_to_primitive(), expected)