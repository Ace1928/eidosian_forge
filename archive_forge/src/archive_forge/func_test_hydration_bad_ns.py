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
def test_hydration_bad_ns(self):
    primitive = {'versioned_object.name': 'MyObj', 'versioned_object.namespace': 'foo', 'versioned_object.version': '1.5', 'versioned_object.data': {'foo': 1}}
    self.assertRaises(exception.UnsupportedObjectError, MyObj.obj_from_primitive, primitive)