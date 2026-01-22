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
def test_compound_clone(self):
    obj = MyCompoundObject()
    obj.foo = [1, 2, 3]
    obj.bar = {'a': 1, 'b': 2, 'c': 3}
    obj.baz = set([1, 2, 3])
    copy = obj.obj_clone()
    self.assertEqual(obj.foo, copy.foo)
    self.assertEqual(obj.bar, copy.bar)
    self.assertEqual(obj.baz, copy.baz)
    copy.foo.append('4')
    copy.bar.update(d='4')
    copy.baz.add('4')
    self.assertEqual([1, 2, 3, 4], copy.foo)
    self.assertEqual({'a': 1, 'b': 2, 'c': 3, 'd': 4}, copy.bar)
    self.assertEqual(set([1, 2, 3, 4]), copy.baz)