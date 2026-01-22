import datetime
from unittest import mock
import warnings
import iso8601
import netaddr
import testtools
from oslo_versionedobjects import _utils
from oslo_versionedobjects import base as obj_base
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields
from oslo_versionedobjects import test
def test_mapping_iterable(self):
    self.assertRaises(ValueError, fields.List(fields.Integer).coerce, None, None, {'a': 1, 'b': 2})