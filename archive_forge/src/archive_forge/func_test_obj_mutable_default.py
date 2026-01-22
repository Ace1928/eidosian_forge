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
def test_obj_mutable_default(self):
    obj = MyObj(context=self.context, foo=123, bar='abc')
    obj.mutable_default = None
    obj.mutable_default.append('s1')
    self.assertEqual(obj.mutable_default, ['s1'])
    obj1 = MyObj(context=self.context, foo=123, bar='abc')
    obj1.mutable_default = None
    obj1.mutable_default.append('s2')
    self.assertEqual(obj1.mutable_default, ['s2'])