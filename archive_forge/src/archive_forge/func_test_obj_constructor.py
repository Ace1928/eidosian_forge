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
def test_obj_constructor(self):
    obj = MyObj(context=self.context, foo=123, bar='abc')
    self.assertEqual(123, obj.foo)
    self.assertEqual('abc', obj.bar)
    self.assertEqual(set(['foo', 'bar']), obj.obj_what_changed())